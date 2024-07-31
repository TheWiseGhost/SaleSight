import React, { Component } from "react";
import { createRoot } from "react-dom/client";
import MyRouter from "./Router";

class App extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <MyRouter />
      </div>
    );
  }
}

const appDiv = document.getElementById("app");
const root = createRoot(appDiv);
root.render(<App />);
