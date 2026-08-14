import globals from "globals";

export default [
  {
    files: ["**/*.js", "**/*.mjs"],
    ignores: ["pkg/**", "node_modules/**"],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
];
