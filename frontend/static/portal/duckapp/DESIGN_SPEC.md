# Journey Buddy Design Specification
## Duck App UI/UX Style Guide

---

## 1. Design Philosophy

**Kid-friendly, playful, approachable.** The design prioritizes:
- Large tap targets for small fingers
- High contrast for readability
- Joyful animations that reward interaction
- Minimal text, heavy use of emoji/icons
- Rounded, soft shapes (no sharp corners)

---

## 2. Color Palette

### Primary Colors
| Name | Hex | Usage |
|------|-----|-------|
| **Duck Yellow** | `#FFD93D` | Duck body, highlights, accents |
| **Happy Orange** | `#FF6B35` | Timer, alerts, energy |
| **Nature Green** | `#4CAF50` | Primary buttons, success, go |
| **Sky Blue** | `#87CEEB` | Backgrounds, calm, water |

### Secondary Colors
| Name | Hex | Usage |
|------|-----|-------|
| **Soft White** | `#FFFFFF` | Cards, panels, text on dark |
| **Warm Beige** | `#F5DEB3` | Neutral buttons, sand |
| **Cozy Purple** | `#5C6BC0` | Sleep/calm actions |
| **Alert Red** | `#CC3333` | Reset, cancel, warning |

### Neutral Colors
| Name | Hex | Usage |
|------|-----|-------|
| **Dark Text** | `#333333` | Primary text |
| **Medium Gray** | `#666666` | Secondary text |
| **Light Gray** | `#888888` | Tertiary text, hints |
| **Background** | `#FAFAFA` | Card backgrounds |
| **Border Light** | `#E8F5E9` | Subtle borders |

---

## 3. Typography

### Font Stack
```css
font-family: 'Segoe UI', system-ui, sans-serif;
```

### Scale
| Element | Size | Weight | Color |
|---------|------|--------|-------|
| **H1 (Header)** | 1.3rem | 600 | `#2D5A27` |
| **H2 (Panel Title)** | 1.4rem | 600 | `#333333` |
| **Timer Display** | 2rem | 700 | `#FF6B35` |
| **Button Text** | 1.1rem | 600 | white |
| **Label** | 0.85rem | 600 | `#333333` |
| **Hint/Subtitle** | 0.9rem | 400 | `#666666` |
| **Small Text** | 0.75rem | 400 | `#888888` |

### Special Effects
```css
/* Header text shadow */
text-shadow: 0 1px 2px rgba(255,255,255,0.5);

/* Timer text shadow */
text-shadow: 0 2px 4px rgba(0,0,0,0.1);

/* Icon outline (for visibility on colored backgrounds) */
text-shadow: 
    -1px -1px 0 #000,
    1px -1px 0 #000,
    -1px 1px 0 #000,
    1px 1px 0 #000,
    0 0 3px rgba(0,0,0,0.5);
```

---

## 4. Spacing & Layout

### Base Unit
`8px` — all spacing should be multiples of 8

### Common Spacing
| Name | Value | Usage |
|------|-------|-------|
| **xs** | 4px | Tight gaps |
| **sm** | 8px | Icon margins |
| **md** | 12px | Button padding, grid gaps |
| **lg** | 20px | Section padding |
| **xl** | 30px | Panel padding |

### Border Radius
| Element | Radius |
|---------|--------|
| Small buttons | 12px |
| Large buttons | 30px (pill) |
| Cards/Panels | 24px |
| Inputs | 12px |
| Theme buttons | 12px |
| Control buttons | 50% (circle) |

---

## 5. Components

### Primary Button
```css
.btn-primary {
    padding: 16px 40px;
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    border-radius: 30px;
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    box-shadow: 0 4px 15px rgba(76,175,80,0.4);
    cursor: pointer;
    transition: transform 0.1s, box-shadow 0.1s;
}
.btn-primary:active {
    transform: scale(0.96);
}
```

### Secondary Button
```css
.btn-secondary {
    background: #f5f5f5;
    color: #666;
}
```

### Control Button (Circle)
```css
.control-btn {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    border: none;
    font-size: 1.8rem;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.control-btn:active {
    transform: scale(0.9);
}
```

### Panel/Card
```css
.panel {
    background: white;
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    text-align: center;
    max-width: 320px;
}
```

### Activity Button (Grid Item)
```css
.activity-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 8px;
    border: 2px solid #E8F5E9;
    border-radius: 16px;
    background: #FAFAFA;
    cursor: pointer;
    transition: all 0.2s;
}
.activity-btn:hover, .activity-btn:active {
    border-color: #4CAF50;
    background: #E8F5E9;
    transform: scale(1.02);
}
.activity-btn .emoji { font-size: 1.8rem; }
.activity-btn .label { font-size: 0.8rem; font-weight: 600; }
.activity-btn .time { font-size: 0.7rem; color: #888; }
```

### Theme Selector Button
```css
.theme-btn {
    width: 50px;
    height: 50px;
    border-radius: 12px;
    border: 3px solid transparent;
    font-size: 1.5rem;
    cursor: pointer;
    transition: all 0.2s;
}
.theme-btn:hover { transform: scale(1.1); }
.theme-btn.selected { 
    border-color: #4CAF50; 
    box-shadow: 0 2px 8px rgba(76,175,80,0.4); 
}
```

---

## 6. Animations

### Button Press
```css
transform: scale(0.96); /* Primary */
transform: scale(0.9);  /* Control buttons */
```

### Pop In (Rewards, Celebrations)
```css
@keyframes popIn {
    from { transform: scale(0); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}
```

### Bounce (Celebration Header)
```css
@keyframes bounce {
    from { transform: translateY(0); }
    to { transform: translateY(-10px); }
}
animation: bounce 0.5s ease infinite alternate;
```

### Timer Celebration
```css
@keyframes timerCelebrate {
    0% { transform: scale(1); color: #FF6B35; }
    10% { transform: scale(1.4) rotate(-3deg); color: #FFD700; }
    20% { transform: scale(1.5) rotate(3deg); color: #FF6B35; }
    30% { transform: scale(1.4) rotate(-2deg); color: #FFD700; }
    40% { transform: scale(1.3) rotate(2deg); color: #FF6B35; }
    60% { transform: scale(1.2) rotate(0deg); color: #FFD700; }
    100% { transform: scale(1) rotate(0deg); color: #FF6B35; }
}
```

### Hold-to-Confirm (Reset Button)
- Circular progress ring fills counter-clockwise
- 2 second hold duration
- Uses `conic-gradient` from 270deg
- Button scales to 0.95 while holding

---

## 7. 3D Style (Three.js)

### Aesthetic
**Blocky / LEGO / Minecraft-inspired**
- All models built from `BoxGeometry` primitives
- No smooth curves — use stacked boxes
- Chunky, toylike proportions

### Materials
```javascript
// Standard objects
new THREE.MeshStandardMaterial({ 
    color: 0xFFD93D,
    roughness: 0.8-0.9 
})

// Shiny/wet (water, ice)
new THREE.MeshStandardMaterial({ 
    color: 0x4A90D9,
    transparent: true,
    opacity: 0.7,
    roughness: 0.1-0.3,
    metalness: 0.1-0.5
})
```

### Lighting
```javascript
// Ambient (soft fill)
new THREE.AmbientLight(0xffffff, 0.6-0.75)

// Directional (sun)
new THREE.DirectionalLight(0xffffff, 0.8)
sun.position.set(10, 20, 10)
sun.castShadow = true
```

### Duck Proportions
| Part | Size (BoxGeometry) |
|------|-----|
| Body | 1.2 × 1.0 × 1.4 |
| Head | 0.9 × 0.9 × 0.9 |
| Beak | 0.4 × 0.25 × 0.5 |
| Wings | 0.2 × 0.6 × 0.8 |
| Feet | 0.4 × 0.15 × 0.6 |
| Eyes | 0.2 × 0.2 × 0.1 |

### Animation Speeds
| Animation | Speed |
|-----------|-------|
| Wing flap (walking) | `sin(time * 2) * 0.2` |
| Foot walk | `sin(time * 3) * 0.1` |
| Idle bob | `sin(time * 0.5) * 0.05` |
| Sleep breathing | `sin(time * 0.8) * 0.03` |

---

## 8. Sound Design

### Characteristics
- Synthesized (Web Audio API)
- Short, cheerful tones
- Triangle/sine waves for melody
- Quick attack, medium decay

### Duck Quack
```javascript
// Two quick descending tones
// 800Hz → 400Hz (0.15s)
// 750Hz → 350Hz (0.15s, 0.18s delay)
```

### Celebration Fanfare
```javascript
// Ascending notes: C5, E5, G5, C6
// Followed by chord of all four
```

### Bell (Wake Up)
```javascript
// Single tone: 1200Hz → 800Hz (0.5s)
```

---

## 9. Biome Themes

### Forest 🌲
- Sky: `#87CEEB`
- Ground: Grass greens `#4a8f3a` to `#3a7f2a`
- Trees: Cone-style pines, dark green
- Animals: Rabbits, squirrels, birds, deer

### Desert 🏜️
- Sky: `#F5DEB3`
- Ground: Browns/tans `#C9A066` to `#A67B5B`
- Plants: Saguaro cacti, tumbleweeds
- Animals: Roadrunners, scorpions, camels, vultures

### Ocean 🌊
- Sky: `#87CEEB`
- Ground: Blues `#2E86AB` to `#1A5276`
- Features: Coral, seaweed, buoys
- Animals: Fish, dolphins, seagulls, sea turtles
- Special: Duck rides in rowboat!

### Winter ❄️
- Sky: `#8BA5B5`
- Ground: Rocky gray-blue `#A8B8C8` to `#778899`
- Features: Blue ice formations, frozen pools
- Trees: Frosted blue-gray pines
- Animals: Penguins, polar bears, owls, arctic foxes

---

## 10. Accessibility

### Touch Targets
- Minimum 44×44px (ideally 70×70px for main controls)
- Adequate spacing between targets

### Visual Feedback
- All interactions have immediate visual response
- Hold-to-confirm for destructive actions
- Animations accompany audio cues

### Color Independence
- Don't rely solely on color to convey meaning
- Use icons/emoji alongside colors
- Timer celebration uses size + motion, not just color

---

## 11. Emoji Usage

### Common UI Emoji
| Emoji | Meaning |
|-------|---------|
| 🦆 | Duck/buddy |
| 😴 | Sleep/pause |
| 🔔 | Wake/resume |
| ↻ | Reset |
| 🚀 | Start/go |
| 🎉 | Celebration |
| ⭐ | Custom/special |
| 🌟 | Achievement |
| 🏆 | Trophy/total |

### Activity Emoji
🪥 Brush Teeth · 👕 Get Dressed · 🧹 Tidy Up · 🍽️ Eat Meal · 📚 Reading · ✏️ Homework · 🧘 Calm Down

### Theme Emoji
🌲 Forest · 🏜️ Desert · 🌊 Ocean · ❄️ Winter

---

*Last updated: February 2026*
*For Journey Buddy / Duck App*
