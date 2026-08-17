/**
 * Background music and per-sign activation sounds.
 *
 * Files are discovered by name, the same way portraits are:
 *
 *   assets/audio/theme.mp3          looped background music
 *   assets/audio/<sign>.mp3         played once when that sign fires
 *
 * Anything missing is simply skipped, so a folder with only a theme in it works
 * fine, as does no folder at all.
 *
 * Browsers refuse to start audio without a user gesture, so nothing here plays
 * until start() is called -- which happens from the Start button's click
 * handler. Calling it any other way gets the play promise rejected and no sound.
 */

const BASE = "../assets/audio";
const EXTENSIONS = ["mp3", "ogg", "m4a", "wav"];

/** Resolve the first extension that actually loads, or null if none do. */
async function findAudio(name) {
  for (const extension of EXTENSIONS) {
    const url = `${BASE}/${name}.${extension}`;
    const element = new Audio();
    const loaded = await new Promise((resolve) => {
      // canplaythrough fires when enough is buffered to play start to finish;
      // error fires for a 404 or an unsupported codec.
      element.addEventListener("canplaythrough", () => resolve(true), { once: true });
      element.addEventListener("error", () => resolve(false), { once: true });
      element.preload = "auto";
      element.src = url;
    });
    if (loaded) return element;
  }
  return null;
}

export class SoundBoard {
  constructor({ signs = [], musicVolume = 0.35, effectVolume = 0.8 } = {}) {
    this.signs = signs;
    this.musicVolume = musicVolume;
    this.effectVolume = effectVolume;
    this.music = null;
    this.effects = new Map();
    this.muted = false;
    this.started = false;
  }

  /** Load whatever exists. Safe to call before any user gesture. */
  async load() {
    this.music = await findAudio("theme");
    if (this.music) {
      this.music.loop = true;
      this.music.volume = this.musicVolume;
    }

    const found = await Promise.all(
      this.signs.map(async (name) => [name, await findAudio(name)])
    );
    for (const [name, element] of found) {
      if (element) {
        element.volume = this.effectVolume;
        this.effects.set(name, element);
      }
    }

    return { music: Boolean(this.music), effects: [...this.effects.keys()] };
  }

  /** Must be called from a user gesture, or the browser blocks playback. */
  async start() {
    this.started = true;
    if (!this.music || this.muted) return;
    try {
      await this.music.play();
    } catch (error) {
      console.warn("Music blocked by the browser's autoplay policy:", error);
    }
  }

  /** Play a sign's activation sound, if it has one. */
  fire(name) {
    if (this.muted || !this.started) return;
    const sound = this.effects.get(name);
    if (!sound) return;
    // Rewind rather than waiting: firing the same sign twice in quick
    // succession should retrigger the sound, not be silently ignored because
    // the element is already playing.
    sound.currentTime = 0;
    sound.play().catch(() => {});
  }

  setMuted(muted) {
    this.muted = muted;
    if (!this.music) return;
    if (muted) this.music.pause();
    else if (this.started) this.music.play().catch(() => {});
  }

  get hasAudio() {
    return Boolean(this.music) || this.effects.size > 0;
  }

  stop() {
    this.music?.pause();
    for (const sound of this.effects.values()) sound.pause();
    this.started = false;
  }
}
