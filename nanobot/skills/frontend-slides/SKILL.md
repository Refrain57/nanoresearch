---
name: frontend-slides
description: Create beautiful HTML presentations from markdown. Use when user asks for slides, presentations, or visual content. Generates standalone HTML files with CSS styling and animations.
---

# Frontend Slides Skill

## Overview

This skill enables you to create beautiful, interactive HTML presentations from markdown content. The output is a single standalone HTML file that can be opened in any browser - no server required.

## When to Use

Use this skill when the user asks for:
- "Create a presentation about X"
- "Make slides for my talk"
- "Generate PPT/slides"
- "I need visual content for my presentation"

## Output Format

**HTML, not PPT.** The output is a self-contained HTML file with:
- Embedded CSS styling
- Keyboard navigation (arrow keys, space)
- Smooth slide transitions
- Responsive design
- Print-friendly (Ctrl+P)

## Style Pressets

12 built-in presets available. Reference `STYLE_PRESETS.md` for full details.

| Preset | Best For |
|--------|----------|
| `minimal` | Clean, professional |
| `dark` | Tech talks, code-heavy |
| `gradient` | Modern, eye-catching |
| `corporate` | Business presentations |
| `playful` | Casual, creative |
| `elegant` | Formal events |
| `tech` | Developer conferences |
| `academic` | Research, thesis defense |
| `vibrant` | Marketing, product launches |
| `monochrome` | Art, design portfolios |
| `nature` | Environmental topics |
| `retro` | Nostalgic themes |

## Usage

1. **Ask user for topic and key points** (or extract from context)
2. **Choose appropriate preset** based on audience/topic
3. **Generate HTML** using the template structure
4. **Save to file** and inform user to open in browser

## Slide Structure

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Presentation Title</title>
  <style>
    /* Base CSS (mandatory) */
    /* + Preset CSS */
  </style>
</head>
<body>
  <div class="slide">...</div>
  <div class="slide">...</div>
  ...
  <script>
    /* Navigation script */
  </script>
</body>
</html>
```

## Content Guidelines

### Title Slide
```html
<div class="slide">
  <h1>Title</h1>
  <p class="subtitle">Subtitle</p>
  <p class="author">Author Name</p>
</div>
```

### Content Slide
```html
<div class="slide">
  <h2>Slide Title</h2>
  <ul>
    <li>Point one</li>
    <li>Point two</li>
  </ul>
</div>
```

### Code Slide
```html
<div class="slide">
  <h2>Code Example</h2>
  <pre><code>
// Your code here
  </code></pre>
</div>
```

### Two-Column Slide
```html
<div class="slide">
  <div class="columns">
    <div class="col">
      <h3>Left Column</h3>
      <p>Content...</p>
    </div>
    <div class="col">
      <h3>Right Column</h3>
      <p>Content...</p>
    </div>
  </div>
</div>
```

## Navigation

- **Arrow keys**: Next/previous slide
- **Space**: Next slide
- **Home/End**: First/last slide
- **F**: Fullscreen mode

## Integration with Research Workflow

When creating research presentations:
1. Use `retrieve_hybrid` to gather relevant claims/insights
2. Structure slides around key findings
3. Include citations on slides
4. Use `academic` or `tech` preset

## Example Workflow

```
User: "Create a 5-slide presentation about 3D Gaussian Splatting"

1. Research the topic (use deep-research skill or retrieve_hybrid)
2. Outline: Title → Intro → Method → Results → Conclusion
3. Choose preset: "tech" (technical topic)
4. Generate HTML with proper structure
5. Save to workspace/presentations/3dgs-slides.html
6. Tell user: "Saved to presentations/3dgs-slides.html - open in browser"
```

## Notes

- No external dependencies required
- Works offline after generation
- Can be shared as single file
- Print to PDF via browser (Ctrl+P)
