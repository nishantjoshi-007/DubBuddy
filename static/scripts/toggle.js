import { setCookie, getCookie, cookiesEnabled } from './cookies.js';

// Toggle theme and store preference
document.querySelector('.theme-switch__checkbox').addEventListener('change', function() {
    const isDarkTheme = this.checked;
    document.body.classList.toggle('dark-theme', isDarkTheme);
    const tosAccepted = getCookie("tosAccepted") || localStorage.getItem("tosAccepted");

    if (tosAccepted === "true" && cookiesEnabled()) {
        setCookie('dark-theme', isDarkTheme, 3650); // Store in cookies
        localStorage.removeItem('dark-theme'); // Remove from localStorage
    } else {
        localStorage.setItem('dark-theme', isDarkTheme); // Store in localStorage
    }
});

// Apply the saved theme on page load
const themeFromCookie = getCookie('dark-theme');
const themeFromLocalStorage = localStorage.getItem('dark-theme');
if (themeFromCookie === 'true' || themeFromLocalStorage === 'true') {
    document.querySelector('.theme-switch__checkbox').checked = true;
    document.body.classList.add('dark-theme');
}

// Form submission handler to convert localStorage to cookies if needed
const form = document.querySelector(".dubbing-form");
if (form) {
    form.addEventListener("submit", function(event) {
        const tosCheck = document.getElementById("FieldsetCheck").checked;
        if (tosCheck) {
            if (localStorage.getItem("tosAccepted") === "false") {
                localStorage.removeItem("tosAccepted");
                setCookie("tosAccepted", "true", 3650); // 10 years
            }
            if (localStorage.getItem('dark-theme')) {
                const isDarkTheme = localStorage.getItem('dark-theme') === 'true';
                setCookie('dark-theme', isDarkTheme, 3650); // Store in cookies
                localStorage.removeItem('dark-theme'); // Remove from localStorage
            }
        }
    });
}