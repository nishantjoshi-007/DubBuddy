// Theme toggle. The preference lives in localStorage only (decisions.md D-31).
//
// The class goes on <html>, and the inline script in index.html has already applied a saved one
// before the first paint. This file only has to put the switch in the right position and keep the
// two in step afterwards. With nothing saved, the operating system decides (prefers-color-scheme)
// and the switch follows it, so the picture on the header always matches the page.

const root = document.documentElement;
const checkbox = document.querySelector(".theme-switch__checkbox");
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function saved() {
  try {
    const value = localStorage.getItem("theme");
    return value === "dark" || value === "light" ? value : "";
  } catch {
    return ""; // private mode, or storage blocked
  }
}

function apply(theme) {
  root.classList.toggle("dark-theme", theme === "dark");
  root.classList.toggle("light-theme", theme === "light");
  if (checkbox) checkbox.checked = theme === "dark";
}

apply(saved() || (systemDark.matches ? "dark" : "light"));

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

// No stored choice: keep following the operating system if it changes while the page is open.
systemDark.addEventListener("change", (event) => {
  if (!saved()) apply(event.matches ? "dark" : "light");
});
