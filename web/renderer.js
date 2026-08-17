/**
 * Canvas drawing for the camera panel: mirrored video, hand skeleton, charge ring.
 *
 * The character portrait is deliberately NOT drawn here. On the desktop side
 * animated GIFs had to be decoded frame by frame and composited by hand, which
 * was most of the render budget. In a browser an <img> animates itself, so the
 * portrait is a plain element beside the canvas and costs nothing.
 */

import { CONNECTIONS } from "./tracker.js";

const HAND_COLORS = { Left: "#ffaa3c", Right: "#7878ff" };
const ENERGY = [255, 150, 60];
const CHARGED = [255, 90, 120];

function mix(from, to, amount) {
  const channel = (index) => Math.round(from[index] + (to[index] - from[index]) * amount);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

/**
 * Draw the mirrored camera frame, filling the canvas while preserving aspect.
 * Mirroring makes the preview behave like a mirror; landmark coordinates are
 * mirrored alongside it so the two stay in step.
 */
export function drawVideo(context, video) {
  const { width, height } = context.canvas;
  context.save();
  context.translate(width, 0);
  context.scale(-1, 1);

  const videoAspect = video.videoWidth / video.videoHeight;
  const canvasAspect = width / height;
  let drawWidth = width;
  let drawHeight = height;
  if (videoAspect > canvasAspect) drawWidth = height * videoAspect;
  else drawHeight = width / videoAspect;

  context.drawImage(
    video,
    (width - drawWidth) / 2,
    (height - drawHeight) / 2,
    drawWidth,
    drawHeight
  );
  context.restore();
}

/** Landmarks arrive in unmirrored video space; flip x to match the drawn frame. */
function toScreen(landmark, width, height) {
  return [(1 - landmark[0]) * width, landmark[1] * height];
}

export function drawHands(context, hands, thickness = 3) {
  const { width, height } = context.canvas;

  for (const [side, landmarks] of Object.entries(hands)) {
    const color = HAND_COLORS[side] ?? "#ffffff";
    const points = landmarks.map((point) => toScreen(point, width, height));

    context.lineWidth = thickness;
    context.strokeStyle = color;
    context.beginPath();
    for (const [start, end] of CONNECTIONS) {
      context.moveTo(points[start][0], points[start][1]);
      context.lineTo(points[end][0], points[end][1]);
    }
    context.stroke();

    for (const [x, y] of points) {
      context.beginPath();
      context.arc(x, y, thickness + 1, 0, Math.PI * 2);
      context.fillStyle = "#ffffff";
      context.fill();
      context.beginPath();
      context.arc(x, y, thickness - 0.5, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    }
  }
}

/** Mean of the palm landmarks, in screen space. */
export function palmCenter(context, hands) {
  const entries = Object.values(hands);
  if (entries.length === 0) return null;
  const { width, height } = context.canvas;
  let x = 0;
  let y = 0;
  for (const landmarks of entries) {
    const [px, py] = toScreen(landmarks[9], width, height);
    x += px;
    y += py;
  }
  return [x / entries.length, y / entries.length];
}

/** The ring that closes as a sign charges -- the visible half of the debounce. */
export function drawChargeRing(context, center, charge) {
  if (!center || charge <= 0.02) return;
  const [x, y] = center;
  const radius = 90 - 40 * charge;
  const color = mix(ENERGY, CHARGED, charge);

  context.save();
  context.strokeStyle = color;
  context.lineCap = "round";

  context.globalAlpha = 0.35;
  context.lineWidth = 2;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.stroke();

  context.globalAlpha = 1;
  context.lineWidth = 5;
  context.shadowColor = color;
  context.shadowBlur = 18;
  context.beginPath();
  context.arc(x, y, radius + 10, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * charge);
  context.stroke();
  context.restore();
}

export function drawLabel(context, text) {
  if (!text) return;
  context.save();
  context.font = "600 26px system-ui, sans-serif";
  const metrics = context.measureText(text);
  context.fillStyle = "rgba(0, 0, 0, 0.6)";
  context.fillRect(16, 16, metrics.width + 28, 46);
  context.fillStyle = "#ffffff";
  context.fillText(text, 30, 46);
  context.restore();
}

/** Size the backing store to the element, accounting for device pixel ratio. */
export function resizeCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}
