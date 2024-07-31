import React, { useState } from "react";
import "../../static/css/index.css";

function StoreModel() {
  const [name, setName] = useState("");
  const [responseMessage, setResponseMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const token = localStorage.getItem("token");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("/api/store_model/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ name }),
      });
      const data = await response.json();
      if (data.auth) {
        navigate("/auth");
      }
      setResponseMessage(data.message);
    } catch (error) {
      setResponseMessage("An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex p-6 overflow-y-hidden w-full bg-white">
      <div className="bg-white rounded-lg flex flex-col w-full">
        <h1 className="text-4xl text-cyan-600 font-abo-one pb-8">
          Store Ai Model
        </h1>
        <form onSubmit={handleSubmit} className="">
          <label className="block text-md font-abo-one text-gray-700 mb-2">
            Model Name:
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter a Model Name"
            className="w-full p-3 border-cyan-300 border-2 rounded-2xl"
          />
          <div className="w-3/4 mx-auto pt-8">
            <button
              type="submit"
              className="w-full font-reem-kufi text-lg text-black py-3 px-4 rounded-2xl bg-cyan-300 focus:outline-none hover:bg-cyan-400"
              disabled={loading}
            >
              {loading ? "Loading..." : "Store Model"}
            </button>
          </div>
        </form>
        {responseMessage && (
          <p className="pt-6 font-reem-kufi text-lg text-center text-black font-normal">
            {responseMessage}
          </p>
        )}
      </div>
    </div>
  );
}

export default StoreModel;
