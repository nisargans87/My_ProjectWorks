// Live Clock
function updateClock() {
    const now = new Date();

    const options = {
        weekday: "short",
        day: "2-digit",
        month: "short",
        year: "numeric"
    };

    const date = now.toLocaleDateString("en-IN", options);
    const time = now.toLocaleTimeString();

    const clock = document.getElementById("clock");

    if (clock) {
        clock.innerHTML = `${date}<br>${time}`;
    }
}

setInterval(updateClock, 1000);
updateClock();


// Image Preview
const input = document.getElementById("fingerprint");
const preview = document.getElementById("preview");
const uploadContent = document.getElementById("uploadContent");

if (input) {

    input.addEventListener("change", function () {

        const file = this.files[0];

        if (!file)
            return;

        if (!file.type.startsWith("image/")) {
            alert("Please upload an image.");
            return;
        }

        const reader = new FileReader();

        reader.onload = function (e) {

            preview.src = e.target.result;

            preview.style.display = "block";

            uploadContent.style.display = "none";

        };

        reader.readAsDataURL(file);

    });

}


// Drag and Drop
const dropArea = document.getElementById("dropArea");

if (dropArea) {

    ["dragenter", "dragover"].forEach(eventName => {

        dropArea.addEventListener(eventName, e => {

            e.preventDefault();

            dropArea.style.background =
                "rgba(56,189,248,.18)";

        });

    });

    ["dragleave", "drop"].forEach(eventName => {

        dropArea.addEventListener(eventName, e => {

            e.preventDefault();

            dropArea.style.background = "";

        });

    });

    dropArea.addEventListener("drop", e => {

        const files = e.dataTransfer.files;

        input.files = files;

        input.dispatchEvent(new Event("change"));

    });

}


// Button Animation
const form = document.getElementById("uploadForm");

const button = document.getElementById("predictBtn");

if (form) {

    form.addEventListener("submit", function () {

        button.disabled = true;

        button.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Scanning Fingerprint...';

    });

}

// Scanner Pulse Animation
const scanner = document.querySelector(".scanner");

if (scanner) {

    setInterval(() => {

        scanner.animate([

            {
                transform: "scale(1)"
            },

            {
                transform: "scale(1.08)"
            },

            {
                transform: "scale(1)"
            }

        ], {

            duration: 1500

        });

    }, 1600);

}