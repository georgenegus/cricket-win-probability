export const FRANCHISES = {
    "Birmingham Phoenix": { color: "#e84e1b", gradient: "linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%)" },
    "London Spirit": { color: "#005baa", gradient: "linear-gradient(135deg, #00c6ff 0%, #0072ff 100%)" },
    "Manchester Originals": { color: "#1c1c1c", gradient: "linear-gradient(135deg, #434343 0%, #000000 100%)" },
    "Northern Superchargers": { color: "#741d70", gradient: "linear-gradient(135deg, #b92b27 0%, #1565C0 100%)" },
    "Oval Invincibles": { color: "#00a698", gradient: "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)" },
    "Southern Brave": { color: "#00b259", gradient: "linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%)" },
    "Trent Rockets": { color: "#ffb81c", gradient: "linear-gradient(135deg, #f7971e 0%, #ffd200 100%)" },
    "Welsh Fire": { color: "#e31837", gradient: "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)" }
};

// Stable hue derived from a team name — same name always yields the same colour.
function hueFromName(name) {
    const s = String(name || "");
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return h % 360;
}

/**
 * Returns { color, gradient } for a team. The 8 Hundred franchises use their real brand
 * colours; every other team (int'l T20 / ODI) gets a distinct, stable gradient derived
 * from its name so no two sides on screen look alike.
 */
export function teamVisual(name) {
    if (name && FRANCHISES[name]) return FRANCHISES[name];
    const hue = hueFromName(name);
    const hue2 = (hue + 28) % 360;
    return {
        color: `hsl(${hue} 68% 55%)`,
        gradient: `linear-gradient(135deg, hsl(${hue} 70% 50%) 0%, hsl(${hue2} 72% 60%) 100%)`,
    };
}
