// Theme toggle. The preference lives in localStorage only (decisions.md D-31).
//
// Three states, not two. "dark" and "light" are a *choice*, stored and honoured for good; "" is no
// choice at all, which is the default and means "whatever the operating system says". A choice is a
// class on <html> that app.css keys off; no choice is no class, so `@media (prefers-color-scheme)`
// decides on its own and keeps deciding — even for a flip this listener never sees (a browser that
// throttles background tabs, a stylesheet that loads late). The inline script in index.html has
// already applied a stored choice before the first paint; this file only has to keep the switch in
// step and write the choice down when it is made.

const root = document.documentElement;
const checkbox = document.querySelector(".theme-switch__checkbox");
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

/** The stored choice, or "" for "follow the operating system". */
function saved() {
  try {
    const value = localStorage.getItem("theme");
    return value === "dark" || value === "light" ? value : "";
  } catch {
    return ""; // private mode, or storage blocked
  }
}

/** Apply "dark", "light", or "" (no class: prefers-color-scheme decides) and match the switch. */
function apply(theme) {
  root.classList.toggle("dark-theme", theme === "dark");
  root.classList.toggle("light-theme", theme === "light");
  if (checkbox) checkbox.checked = theme === "dark" || (theme === "" && systemDark.matches);
}

apply(saved());

if (checkbox) {
  checkbox.addEventListener("change", () => {
    const theme = checkbox.checked ? "dark" : "light";
    apply(theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // the page still changes; only the memory of it is lost
    }
  });
}

// No stored choice: follow the operating system while the page is open. With one stored, the flip
// is ignored — the person has said what they want and a change of room lighting does not undo it.
systemDark.addEventListener("change", () => {
  if (!saved()) apply("");
});
