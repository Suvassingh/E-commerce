
const express = require("express");
const app = express();

// Example route
app.get("/api/hello", (req, res) => {
  res.json({ message: "Hello from Node.js on Vercel!" });
});

// Export app instead of listening
module.exports = app;