const spinner = document.getElementById("spinner");
const videoContainer = document.getElementById("video-container");
const translatedVideo = document.getElementById("translated-video");
const downloadButton = document.getElementById("download-button");

// long-polling to check the status of the video processing
function checkStatus() {
    fetch('/process-status')
        .then(response => response.json())
        .then(data => {
            console.log(data);
            if (data.status === "completed") {
                spinner.style.display = 'none';
                videoContainer.style.display = 'block';
                downloadButton.style.display = 'block';

                // Set the source of the video
                translatedVideo.src = data.final_video;
                translatedVideo.load();

                clearInterval(statusInterval); // Stop polling
            }
        })
        .catch(error => console.error('Error:', error));
}

// Start polling every 5 seconds
const statusInterval = setInterval(checkStatus, 5000);

// Get the download button and add an event listener to it
downloadButton.addEventListener("click", async function (event) {
    event.preventDefault(); // Prevent default form submission
    const response = await fetch("/download-video");
    if (!response.ok) {
        alert("Error downloading translated video.");
        return;
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = "translated_video.mp4";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
});