document.getElementById('contactForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const response = await fetch('/contact-us', {
        method: 'POST',
        body: formData
    });

    const alertContainer = document.getElementById('alert-container');
    if (response.ok) {
        // const result = await response.json();
        showAlert("The form has been submitted successfully. We will contact you regarding your inquiry on provided email.", 'success');
        form.reset();
    } else {
        showAlert('There was an error submitting the form. Please try again.', 'danger');
    }

    // Reload the page
    setTimeout(() => {
        window.location.reload();
    }, 3000);
});

function showAlert(message, type) {
    const alertContainer = document.getElementById('alert-container');
    alertContainer.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
}


// Maintain a list of files
let uploadedFiles = [];

// Event listener for file input change
document.getElementById('file').addEventListener('change', function(event) {
    // Append selected files to the list
    for (let file of event.target.files) {
        uploadedFiles.push(file);
    }

    // Update the display
    updateFileDisplay();
    
    // Update the form data
    const fileInput = document.getElementById('file');
    const dataTransfer = new DataTransfer();
    uploadedFiles.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
});

function updateFileDisplay() {
    const fileUploadText = document.getElementById('file-upload-text');
    fileUploadText.innerHTML = ""; // Clear previous content

    if (uploadedFiles.length === 0) {
        fileUploadText.innerHTML = "<p>Click Here to Browse Files</p>";
    } else {
        uploadedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span>${file.name}</span>
                <button type="button" class="remove-file-btn" data-index="${index}">Remove</button>
            `;
            fileUploadText.appendChild(fileItem);
        });

        // Add event listeners for remove buttons
        document.querySelectorAll('.remove-file-btn').forEach(button => {
            button.addEventListener('click', function(event) {
                event.stopPropagation(); // Stop the event from propagating
                const index = parseInt(event.target.getAttribute('data-index'));
                removeFile(index);
            });
        });
    }
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateFileDisplay();

    // Update the form data
    const fileInput = document.getElementById('file');
    const dataTransfer = new DataTransfer();
    uploadedFiles.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
}