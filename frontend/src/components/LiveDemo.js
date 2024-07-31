import React from "react";

const LiveDemo = () => {
  return (
    <div className="flex flex-col min-h-screen pt-20" id="demo">
      <h1 className="font-abo-one text-4xl text-center">Live Demo</h1>
      <div className="relative w-4/5 mx-auto mt-6">
        <iframe
          src="https://www.loom.com/embed/8d3e28987ece46b4bfe3c4fdc0e8bd15"
          className="h-96 w-full"
        ></iframe>
      </div>
    </div>
  );
};

export default LiveDemo;
