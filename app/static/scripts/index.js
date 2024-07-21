document.addEventListener('DOMContentLoaded', function() {
    fetch('static/languages.json')
        .then(response => response.json())
        .then(data => {
            const fromLanguages = data.from_languages;
            const toLanguages = data.to_languages;
            const fromLangSelect = document.getElementById('SelectLanguage');
            const toLangSelect = document.getElementById('SelectDubLanguage');

            fromLanguages.forEach(language => {
                const option = document.createElement('option');
                option.textContent = language;
                option.value = language;
                fromLangSelect.appendChild(option);
            });

            toLanguages.forEach(language => {
                const option = document.createElement('option');
                option.textContent = language;
                option.value = language;
                toLangSelect.appendChild(option);
            });
            // Set default selected options
            fromLangSelect.value = "English"; // Set default for 'from' language
            toLangSelect.value = "Spanish"; // Set default for 'to' language
        })
        .catch(error => console.error('Error loading languages:', error));
});