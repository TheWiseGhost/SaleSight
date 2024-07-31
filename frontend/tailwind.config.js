/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Include JS, JSX, TS, TSX files under src/
  ],
  theme: {
    extend: {
      fontFamily: {
        "reem-kufi": ["Reem Kufi", "sans-serif"],
        "abo-one": ["Aoboshi One", "sans-serif"],
      },
      colors: {
        mygreen: "#8BFF63",
        lightgreen: "#B2FF97",
        brightgreen: "#34CA00",
      },
    }, // Optionally extend Tailwind's default theme here
  },
  plugins: [require("daisyui")], // Optionally add plugins here
};
