const video = document.getElementById("video");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

// Capture frame + send to backend
function captureAndSend() {

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const image = canvas.toDataURL("image/jpeg");

    const text = document.getElementById("textInput").value;

    fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        text: text,
        image: image
    })
})
.then(res => res.json())
.then(data => {
    document.getElementById("finalEmotion").innerText =
        data.final_emotion;

    document.getElementById("finalConfidence").innerText =
        data.final_confidence;

    document.getElementById("decision").innerText =
        data.decision;
})
.catch(err => {
    console.error(err);
    alert("Something went wrong");
});
}