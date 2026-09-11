const BACKEND_HTTP = "http://localhost:8000";
const BACKEND_WS = "ws://localhost:8000/ws/stream";
const FRAME_INTERVAL_MS = 250;
const JPEG_QUALITY = 0.85;
const CAMERA_CONSTRAINTS = { width: { ideal: 1280 }, height: { ideal: 720 } };

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d");

const statusEl = document.getElementById("status");
const nameInput = document.getElementById("nameInput");
const enrollBtn = document.getElementById("enrollBtn");
const enrollMsg = document.getElementById("enrollMsg");
const peopleList = document.getElementById("peopleList");

let ws = null;
let sendTimer = null;

function setStatus(connected) {
  statusEl.textContent = connected ? "Conectado" : "Desconectado";
  statusEl.className = "status " + (connected ? "status--on" : "status--off");
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}

function captureFrameDataUrl() {
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

function drawDetections(faces) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!video.videoWidth || !video.videoHeight) return;

  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;

  for (const face of faces) {
    const [x, y, w, h] = face.box;
    const bx = x * scaleX;
    const by = y * scaleY;
    const bw = w * scaleX;
    const bh = h * scaleY;

    const known = Boolean(face.name);
    overlayCtx.strokeStyle = known ? "#7bffa0" : "#ff8080";
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(bx, by, bw, bh);

    const label = known ? `${face.name} (${face.score})` : "Desconocido";
    overlayCtx.font = "14px system-ui, sans-serif";
    const textWidth = overlayCtx.measureText(label).width;
    overlayCtx.fillStyle = known ? "#7bffa0" : "#ff8080";
    overlayCtx.fillRect(bx, Math.max(0, by - 20), textWidth + 10, 20);
    overlayCtx.fillStyle = "#14161a";
    overlayCtx.fillText(label, bx + 5, Math.max(14, by - 5));
  }
}

function connectWebSocket() {
  ws = new WebSocket(BACKEND_WS);

  ws.onopen = () => {
    setStatus(true);
    sendTimer = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ type: "frame", image: captureFrameDataUrl() }));
    }, FRAME_INTERVAL_MS);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "detections") {
      drawDetections(data.faces);
    }
  };

  ws.onclose = () => {
    setStatus(false);
    clearInterval(sendTimer);
    setTimeout(connectWebSocket, 1500);
  };

  ws.onerror = () => {
    ws.close();
  };
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: CAMERA_CONSTRAINTS });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  resizeOverlay();
  window.addEventListener("resize", resizeOverlay);
}

function showEnrollMessage(text, ok) {
  enrollMsg.textContent = text;
  enrollMsg.className = "msg " + (ok ? "ok" : "error");
}

async function loadPeople() {
  const res = await fetch(`${BACKEND_HTTP}/people`);
  const people = await res.json();

  peopleList.innerHTML = "";
  if (people.length === 0) {
    peopleList.innerHTML = '<li class="empty">Todavia no hay nadie registrado.</li>';
    return;
  }

  for (const person of people) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${person.name}</span>`;
    const delBtn = document.createElement("button");
    delBtn.textContent = "Eliminar";
    delBtn.onclick = async () => {
      await fetch(`${BACKEND_HTTP}/people/${person.id}`, { method: "DELETE" });
      loadPeople();
    };
    li.appendChild(delBtn);
    peopleList.appendChild(li);
  }
}

enrollBtn.addEventListener("click", async () => {
  const name = nameInput.value.trim();
  if (!name) {
    showEnrollMessage("Escribe un nombre primero.", false);
    return;
  }

  enrollBtn.disabled = true;
  try {
    const res = await fetch(`${BACKEND_HTTP}/enroll`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, image: captureFrameDataUrl() }),
    });
    const data = await res.json();

    if (!res.ok) {
      showEnrollMessage(data.detail || "No se pudo registrar el rostro.", false);
    } else {
      showEnrollMessage(`"${data.name}" registrado correctamente.`, true);
      nameInput.value = "";
      loadPeople();
    }
  } catch (err) {
    showEnrollMessage("Error de conexion con el backend.", false);
  } finally {
    enrollBtn.disabled = false;
  }
});

(async function init() {
  await startCamera();
  connectWebSocket();
  loadPeople();
})();
