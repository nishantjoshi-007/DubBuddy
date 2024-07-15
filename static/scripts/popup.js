import { setCookie, getCookie, eraseCookie } from './cookies.js';

document.addEventListener("DOMContentLoaded", function() {
    const tosModal = document.getElementById("tos-modal");
    const acceptButton = document.getElementById("accept-tos");
    const declineButton = document.getElementById("decline-tos");
    const closeModalButton = document.getElementById("close-modal");

    // Check if the TOS has already been accepted or declined
    const tosAccepted = getCookie("tosAccepted") || localStorage.getItem("tosAccepted");
    if (!tosAccepted) {
        // Show the modal if TOS not yet accepted or declined
        tosModal.classList.remove("hidden");
    }

    if (getCookie("tosAccepted")) {
        document.getElementById("FieldsetCheck").checked = true;
    } else {
        document.getElementById("FieldsetCheck").checked = false;
    }

    acceptButton.addEventListener("click", function() {
        // Set cookie to remember acceptance
        setCookie("tosAccepted", "true", 3650); // 10 years
        localStorage.removeItem("tosAccepted");
        // Check the Terms of Service checkbox
        document.getElementById("FieldsetCheck").checked = true;
        tosModal.classList.add("hidden");
    });

    declineButton.addEventListener("click", function() {
        eraseCookie("tosAccepted");
        eraseCookie("dark-theme");
        // Set localStorage to remember declination
        localStorage.setItem("tosAccepted", "false");
        // Uncheck the Terms of Service checkbox
        document.getElementById("FieldsetCheck").checked = false;
        tosModal.classList.add("hidden");
    });

    closeModalButton.addEventListener("click", function() {
        tosModal.classList.add("hidden");
    });

    // Form submission handler
    const form = document.querySelector(".dubbing-form");
    if (form) {
        form.addEventListener("submit", function(event) {
            const tosCheck = document.getElementById("FieldsetCheck").checked;
            if (tosCheck) {
                // If the TOS checkbox is checked, switch from localStorage to cookies if needed
                if (localStorage.getItem("tosAccepted") === "false") {
                    localStorage.removeItem("tosAccepted");
                    setCookie("tosAccepted", "true", 3650); // 10 years
                }
            }
        });
    }

// Add event listener to the TOS link
if (tosLink) {
    tosLink.addEventListener("click", function(event) {
        event.preventDefault();
        tosModal.classList.remove("hidden");
        });
    }
});