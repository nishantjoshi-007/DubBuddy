"""Translation behind one interface, with Argos as the offline implementation (flow.md B4.4, D-36).

Argos ships only `xx↔en` packages, so every non-English pair goes through English: install `src→en` and
`en→dst` and `Language.get_translation()` hands back the composed pivot.  Installs are lazy (nobody wants
100 packages on disk) and serialised with a lock, because `MAX_CONCURRENT_JOBS` may be > 1 and
`install_from_path` unzips into a shared directory.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from respeak.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from argostranslate.translate import ITranslation

log = logging.getLogger(__name__)

PIVOT = "en"

#: Serialises package-index updates and installs across worker threads (Argos' own lock does not cover the
#: "look, then install" window).
_install_lock = threading.Lock()
_index_updated = False


class TranslationError(RuntimeError):
    """No usable route between two languages, or a translation call failed. Never signalled by None."""


@runtime_checkable
class Translator(Protocol):
    """What the pipeline needs from a translation engine (an LLM backend can implement this later)."""

    def ensure_pair(self, src: str, dst: str) -> None:
        """Make `src`→`dst` usable, downloading whatever is missing. Raise TranslationError if impossible."""
        ...

    def translate(self, texts: list[str], src: str, dst: str) -> list[str]:
        """Translate every string, 1:1 with the input list and in the same order."""
        ...


def normalize_code(code: str) -> str:
    """'zh-CN' → 'zh', 'PT_br' → 'pt'. ISO-639-1 as in `lang_codes.NAME_TO_CODE`."""
    base = str(code or "").strip().lower().replace("_", "-").split("-")[0]
    if len(base) != 2 or not base.isalpha():
        raise TranslationError(f"not an ISO-639-1 language code: {code!r}")
    return base


class ArgosTranslator:
    """Argos Translate, pivoting through English. Safe to share between worker threads."""

    name = "argos"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._routes: dict[tuple[str, str], ITranslation] = {}
        self._routes_lock = threading.Lock()

    # -- interface ---------------------------------------------------------------------------------

    def ensure_pair(self, src: str, dst: str) -> None:
        src, dst = normalize_code(src), normalize_code(dst)
        self._route(src, dst)

    def translate(self, texts: list[str], src: str, dst: str) -> list[str]:
        src, dst = normalize_code(src), normalize_code(dst)
        if not texts:
            return []
        route = self._route(src, dst)
        out: list[str] = []
        for index, text in enumerate(texts):
            source = (text or "").strip()
            if not source:
                out.append("")
                continue
            try:
                result = (route.translate(source) or "").strip()
            except Exception as exc:
                raise TranslationError(f"{src}->{dst} failed on segment {index}: {exc}") from exc
            if not result:
                log.warning(
                    "%s->%s returned nothing for segment %d; keeping the source text", src, dst, index
                )
                result = source
            out.append(result)
        return out

    # -- optional helper ---------------------------------------------------------------------------

    def supported_targets(self, src: str) -> set[str]:
        """Codes reachable from `src` per the package index, without installing anything.

        Reads the local index only; it is refreshed the first time an install needs the network.
        """
        src = normalize_code(src)
        pairs = self._known_pairs()
        if src != PIVOT and (src, PIVOT) not in pairs:
            return set()
        targets = {dst for (from_code, dst) in pairs if from_code == PIVOT}
        targets.discard(src)
        return targets

    # -- internals ---------------------------------------------------------------------------------

    def _route(self, src: str, dst: str) -> ITranslation:
        if src == dst:
            raise TranslationError(
                f"source and target language are both {src!r}; pick a different target language"
            )
        with self._routes_lock:
            cached = self._routes.get((src, dst))
        if cached is not None:
            return cached

        self._install_hops(src, dst)
        route = self._lookup(src, dst)
        if route is None:
            raise TranslationError(
                f"no Argos route from {src!r} to {dst!r}: "
                f"a {src}->{PIVOT} and/or {PIVOT}->{dst} package is missing and could not be installed"
            )
        with self._routes_lock:
            self._routes[(src, dst)] = route
        return route

    def _install_hops(self, src: str, dst: str) -> None:
        """Install the pivot hops this pair needs, if they are not there already."""
        hops = [(code, PIVOT) for code in (src,) if code != PIVOT]
        hops += [(PIVOT, code) for code in (dst,) if code != PIVOT]
        with _install_lock:
            missing = [hop for hop in hops if hop not in self._installed_pairs()]
            if not missing:
                return
            self._update_index_once()
            available = self._available_packages()
            for from_code, to_code in missing:
                pkg = available.get((from_code, to_code))
                if pkg is None:
                    raise TranslationError(
                        f"Argos has no {from_code}->{to_code} package "
                        f"(needed to translate {src} -> {dst} through {PIVOT})"
                    )
                log.info("installing Argos package %s->%s", from_code, to_code)
                try:
                    from argostranslate import package as argos_package

                    path = pkg.download()
                    argos_package.install_from_path(path)
                except Exception as exc:
                    raise TranslationError(
                        f"could not install the Argos {from_code}->{to_code} package: {exc}"
                    ) from exc

    @staticmethod
    def _update_index_once() -> None:
        """Refresh the remote index at most once per process; offline, the shipped/cached copy is used."""
        global _index_updated
        if _index_updated:
            return
        from argostranslate import package as argos_package

        try:
            argos_package.update_package_index()
        except Exception as exc:  # pragma: no cover - argos already swallows most of these
            log.warning("could not refresh the Argos package index (offline?): %s", exc)
        _index_updated = True

    @staticmethod
    def _installed_pairs() -> set[tuple[str, str]]:
        from argostranslate import package as argos_package

        return {
            (pkg.from_code, pkg.to_code)
            for pkg in argos_package.get_installed_packages()
            if getattr(pkg, "type", "translate") == "translate"
        }

    @staticmethod
    def _available_packages() -> dict[tuple[str, str], object]:
        from argostranslate import package as argos_package

        try:
            packages = argos_package.get_available_packages()
        except Exception as exc:
            log.warning("could not read the Argos package index: %s", exc)
            return {}
        return {
            (pkg.from_code, pkg.to_code): pkg
            for pkg in packages
            if getattr(pkg, "type", "translate") == "translate"
        }

    @classmethod
    def _known_pairs(cls) -> set[tuple[str, str]]:
        return cls._installed_pairs() | set(cls._available_packages())

    @staticmethod
    def _lookup(src: str, dst: str) -> ITranslation | None:
        """Argos builds the composed src->en->dst translation when loading installed languages."""
        from argostranslate import translate as argos_translate

        languages = {lang.code: lang for lang in argos_translate.get_installed_languages()}
        from_lang = languages.get(src)
        to_lang = languages.get(dst)
        if from_lang is None or to_lang is None:
            return None
        return from_lang.get_translation(to_lang)
