// Theme toggle. Preference lives in localStorage only (decisions.md D-31).
const checkbox = document.querySelector('.theme-switch__checkbox');
const saved = localStorage.getItem('theme');
if (saved === 'dark') { document.body.classList.add('dark-theme'); if (checkbox) checkbox.checked = true; }
if (saved === 'light') { document.body.classList.add('light-theme'); }
if (checkbox) {
  checkbox.addEventListener('change', function () {
    const dark = this.checked;
    document.body.classList.toggle('dark-theme', dark);
    document.body.classList.toggle('light-theme', !dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light');
  });
}
