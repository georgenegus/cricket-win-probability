export const FORMATS = {
    t20: { id: "t20", label: "T20", overs: 20, ballsPerOver: 6, totalBalls: 120, ppEndBall: 36, deathStartBall: 90, parScore: 160, unit: "overs" },
    od: { id: "od", label: "ODI", overs: 50, ballsPerOver: 6, totalBalls: 300, ppEndBall: 60, deathStartBall: 240, parScore: 260, unit: "overs" },
    hnd: { id: "hnd", label: "The Hundred", overs: 20, ballsPerOver: 5, totalBalls: 100, ppEndBall: 25, deathStartBall: 75, parScore: 130, unit: "balls" },
    t20w: { id: "t20w", label: "T20 (Women)", overs: 20, ballsPerOver: 6, totalBalls: 120, ppEndBall: 36, deathStartBall: 90, parScore: 125, unit: "overs" },
    odw: { id: "odw", label: "ODI (Women)", overs: 50, ballsPerOver: 6, totalBalls: 300, ppEndBall: 60, deathStartBall: 240, parScore: 221, unit: "overs" },
    hndw: { id: "hndw", label: "The Hundred (Women)", overs: 20, ballsPerOver: 5, totalBalls: 100, ppEndBall: 25, deathStartBall: 75, parScore: 127, unit: "balls" },
};

export const ballsToLabel = (f, ballsBowled) =>
    f.unit === "balls" ? `${ballsBowled}/${f.totalBalls} balls`
                       : `${Math.floor(ballsBowled / f.ballsPerOver)}.${ballsBowled % f.ballsPerOver} ov`;

export const phaseOf = (f, ballsBowled) =>
    ballsBowled < f.ppEndBall ? "powerplay" : ballsBowled >= f.deathStartBall ? "death" : "middle";
