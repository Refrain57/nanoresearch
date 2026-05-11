# Style Presets Reference

## Mandatory Base CSS

Every presentation must include this base CSS:

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.slide {
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 4rem;
  position: absolute;
  top: 0;
  left: 100%;
  transition: left 0.5s ease;
}
.slide.active { left: 0; }
.slide.done { left: -100%; }
```

## Viewport Rules

- Each slide: `width: 100vw; height: 100vh;`
- Content should fit within viewport
- Use `font-size: clamp(min, preferred, max)` for responsive text

---

## Preset: minimal

Clean, professional look. Good for general use.

```css
body { background: #fff; color: #333; }
h1 { font-size: clamp(2rem, 5vw, 4rem); margin-bottom: 1rem; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); margin-bottom: 1.5rem; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: disc; padding-left: 2rem; }
.subtitle { font-size: 1.5rem; color: #666; }
.author { margin-top: 2rem; color: #888; }
```

---

## Preset: dark

Dark theme, ideal for tech talks and code-heavy presentations.

```css
body { background: #1a1a2e; color: #eee; }
h1 { font-size: clamp(2rem, 5vw, 4rem); color: #00d9ff; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); color: #00d9ff; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: none; padding-left: 0; }
li::before { content: "▸ "; color: #00d9ff; }
pre { background: #0d0d1a; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }
code { font-family: 'Fira Code', monospace; color: #7dd3fc; }
.subtitle { color: #888; }
```

---

## Preset: gradient

Modern gradient background, eye-catching.

```css
body {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
}
h1 { font-size: clamp(2rem, 5vw, 4rem); text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: none; }
li::before { content: "✦ "; }
```

---

## Preset: corporate

Professional business style.

```css
body { background: #f8f9fa; color: #212529; }
h1 { font-size: clamp(2rem, 5vw, 4rem); color: #0d6efd; border-bottom: 3px solid #0d6efd; padding-bottom: 0.5rem; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); color: #0d6efd; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: square; padding-left: 2rem; }
.subtitle { color: #6c757d; }
```

---

## Preset: playful

Fun, casual style with rounded elements.

```css
body { background: #fff9e6; color: #333; }
h1 { font-size: clamp(2rem, 5vw, 4rem); color: #ff6b6b; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); color: #4ecdc4; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: "→ "; padding-left: 2rem; }
.slide { border-radius: 2rem; }
```

---

## Preset: elegant

Sophisticated, formal style.

```css
body { background: #fafafa; color: #2c2c2c; font-family: 'Georgia', serif; }
h1 { font-size: clamp(2rem, 5vw, 4rem); font-weight: 300; letter-spacing: 0.1em; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); font-weight: 300; border-bottom: 1px solid #ccc; padding-bottom: 0.5rem; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.8; }
ul { list-style: none; padding-left: 0; }
li::before { content: "— "; color: #888; }
```

---

## Preset: tech

Developer/tech conference style.

```css
body { background: #0f0f23; color: #cccccc; font-family: 'JetBrains Mono', 'Fira Code', monospace; }
h1 { font-size: clamp(2rem, 5vw, 4rem); color: #00ff41; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); color: #00ff41; }
p, li { font-size: clamp(0.9rem, 1.8vw, 1.3rem); line-height: 1.6; }
ul { list-style: "> "; padding-left: 2rem; }
pre { background: #1a1a3e; padding: 1rem; border: 1px solid #00ff41; }
code { color: #00ff41; }
a { color: #00d4ff; }
```

---

## Preset: academic

Research/academic presentation style.

```css
body { background: #fff; color: #1a1a1a; font-family: 'Latin Modern Roman', 'Times New Roman', serif; }
h1 { font-size: clamp(1.8rem, 4vw, 3rem); font-weight: normal; }
h2 { font-size: clamp(1.3rem, 3vw, 2rem); font-weight: normal; }
p, li { font-size: clamp(0.9rem, 1.8vw, 1.2rem); line-height: 1.7; }
ul { list-style: decimal; padding-left: 2rem; }
.citation { font-size: 0.8rem; color: #666; font-style: italic; }
.figure { font-size: 0.9rem; color: #444; text-align: center; }
```

---

## Preset: vibrant

Bold colors, marketing/product launches.

```css
body { background: linear-gradient(180deg, #ff0844 0%, #ffb199 100%); color: #fff; }
h1 { font-size: clamp(2.5rem, 6vw, 5rem); font-weight: 900; }
h2 { font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 700; }
p, li { font-size: clamp(1.1rem, 2.2vw, 1.6rem); line-height: 1.5; }
ul { list-style: "★ "; }
```

---

## Preset: monochrome

Black and white, artistic.

```css
body { background: #000; color: #fff; }
h1 { font-size: clamp(2rem, 5vw, 4rem); font-weight: 100; letter-spacing: 0.2em; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); font-weight: 100; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.8; font-weight: 300; }
ul { list-style: "○ "; }
```

---

## Preset: nature

Green/earth tones, environmental topics.

```css
body { background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); color: #fff; }
h1 { font-size: clamp(2rem, 5vw, 4rem); }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: "🌿 "; }
```

---

## Preset: retro

Vintage/nostalgic style.

```css
body { background: #f4e4bc; color: #3d2914; font-family: 'Courier New', monospace; }
h1 { font-size: clamp(2rem, 5vw, 4rem); text-transform: uppercase; letter-spacing: 0.1em; }
h2 { font-size: clamp(1.5rem, 4vw, 3rem); text-transform: uppercase; }
p, li { font-size: clamp(1rem, 2vw, 1.5rem); line-height: 1.6; }
ul { list-style: "► "; }
```

---

## Navigation Script (Required)

```javascript
const slides = document.querySelectorAll('.slide');
let current = 0;
slides[0].classList.add('active');

function show(n) {
  slides[current].classList.remove('active');
  slides[current].classList.add(current < n ? 'done' : '');
  slides[n].classList.remove('done');
  slides[n].classList.add('active');
  current = n;
}

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ') show(Math.min(current + 1, slides.length - 1));
  if (e.key === 'ArrowLeft') show(Math.max(current - 1, 0));
  if (e.key === 'Home') show(0);
  if (e.key === 'End') show(slides.length - 1);
  if (e.key === 'f' || e.key === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
});
```
