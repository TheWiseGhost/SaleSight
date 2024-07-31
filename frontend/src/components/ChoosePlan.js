import React, { useState } from "react";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";

const ChoosePlan = () => {
  const [activePlan, setActivePlan] = useState("one-time");

  return (
    <div className="max-w-lg pl-12 mx-auto p-8 bg-white rounded-lg">
      <h1 className="text-center text-4xl font-abo-one mb-6">Sale Sight</h1>
      <div className="flex justify-center mb-8">
        <button
          className={`px-4 py-2 mx-2 font-reem-kufi text-lg rounded-2xl ${
            activePlan === "monthly" ? "border-2 border-lightgreen" : "bg-white"
          }`}
          onClick={() => setActivePlan("monthly")}
        >
          <span className="text-green-500 text-xl">$9.99 </span> <br />{" "}
          recurring monthly
        </button>
        <button
          className={`px-4 py-2 mx-2 font-reem-kufi text-lg rounded-2xl ${
            activePlan === "one-time"
              ? "border-2 border-lightgreen"
              : "bg-white"
          }`}
          onClick={() => setActivePlan("one-time")}
        >
          <span className="text-green-500 text-xl">$49.99 </span> <br />
          one time payment
        </button>
      </div>
      <ul className="mb-8 space-y-4 font-reem-kufi">
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-mygreen" />
          Unlimited AI Predictions
        </li>
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-yellow-300" />
          Unlimited AI Models
        </li>
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-cyan-300" />
          Unlimited Data Storage
        </li>
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-mygreen" />
          Ability to add as many features or rows to the Data Table
        </li>
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-yellow-300" />
          Key Data Insights for adjustments
        </li>
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-cyan-300" />7
          Key Data Feature Columns
        </li>
        <li className="flex items-center mb-2">
          <CheckCircleIcon fontSize="medium" className="mr-2 text-mygreen" />
          Easy and fast to CSV export and import from Excel
        </li>
      </ul>
      <div className="flex justify-center">
        <button className="px-8 py-4 bg-lightgreen text-black font-abo-one rounded-2xl hover:bg-mygreen transition duration-200">
          Choose Plan
        </button>
      </div>
    </div>
  );
};

export default ChoosePlan;
