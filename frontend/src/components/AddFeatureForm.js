import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const AddFeatureForm = () => {
  const [selectedTable, setSelectedTable] = useState("Offers");
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);
    setError(null);

    const data = {
      table: selectedTable,
      text: inputText,
    };

    try {
      const response = await fetch("/api/add_feature/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(data),
      });

      const responseData = await response.json();
      if (responseData.auth) {
        navigate("/auth");
      }
      if (response.status === 201) {
        setResult("Entry added successfully");
        setInputText("");
        window.location.reload();
      } else {
        setError(responseData.error || "Failed to add entry");
      }
    } catch (error) {
      console.error("Error adding entry:", error);
      setError("Error adding entry");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white rounded-md">
      <h2 className="text-4xl text-brightgreen font-abo-one pb-8">
        Add Feature
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="mb-4">
          <label className="block text-md font-abo-one text-gray-700">
            Select Table:
          </label>
          <select
            value={selectedTable}
            onChange={(e) => setSelectedTable(e.target.value)}
            className="form-select font-reem-kufi border-2 mt-1 block w-full px-3 py-2 rounded-2xl border-lightgreen shadow-sm focus:border-mygreen"
          >
            <option value="Offers">Offers</option>
            <option value="Responses">Responses</option>
            <option value="Locations">Locations</option>
            <option value="Sale_Products">Sale Products</option>
            <option value="Lead_Gens">Lead Gens</option>
            <option value="Results">Results</option>
            <option value="Funnel_Stages">Funnel_Stages</option>
          </select>
        </div>
        <div className="mb-4">
          <label className="block text-md font-abo-one text-gray-700">
            Enter Text:
          </label>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            required
            className="font-reem-kufi form-input border-2 mt-1 block w-full px-3 py-2 rounded-2xl border-lightgreen shadow-sm focus:border-mygreen"
          />
        </div>
        <div className="flex items-center justify-between">
          <div className="w-3/4 mx-auto pt-4">
            <button
              type="submit"
              className="w-full font-reem-kufi text-lg text-black py-3 px-4 rounded-2xl bg-lightgreen focus:outline-none hover:bg-mygreen"
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Add Feature"}
            </button>
          </div>
        </div>
      </form>
      {result && (
        <div className="mt-4 text-green-500">
          <h3 className="text-xl font-bold">Success:</h3>
          <pre className="bg-gray-100 p-4 rounded-md">{result}</pre>
        </div>
      )}
      {error && (
        <div className="mt-4 text-red-500">
          <h3 className="text-xl font-bold">Error:</h3>
          <pre className="bg-gray-100 p-4 rounded-md">{error}</pre>
        </div>
      )}
    </div>
  );
};

export default AddFeatureForm;
