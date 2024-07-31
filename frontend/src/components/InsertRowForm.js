import React, { useState, useEffect } from "react";
import "../../static/css/index.css";

const InsertFormRow = () => {
  const [formData, setFormData] = useState({
    response: "",
    location: "",
    lead_gen: "",
    offer: "",
    sale_product: "",
    funnel_stage: "",
    result: "",
  });

  const [dropdownOptions, setDropdownOptions] = useState({
    responses: [],
    locations: [],
    lead_Gens: [],
    offers: [],
    sale_Products: [],
    funnel_Stages: [],
    results: [],
  });

  const translator = {
    response: "Responses",
    location: "Locations",
    lead_gen: "Lead_Gens",
    offer: "Offers",
    sale_product: "Sale_Products",
    funnel_stage: "Funnel_Stages",
    result: "Results",
  };

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true); // Track loading state
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true); // Set loading to true before starting the fetch
      const token = localStorage.getItem("token"); // Retrieve the token from localStorage

      try {
        const response = await fetch("/api/get_feature_values/", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // Include JWT in headers
          },
        });
        if (response.auth) {
          navigate("/auth");
        }
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Fetched dropdown options:", data);
        setDropdownOptions(data);
      } catch (error) {
        console.error("Error fetching dropdown options:", error);
        setError("Error fetching dropdown options");
      } finally {
        setLoading(false); // Ensure loading is set to false after fetch attempt
      }
    };

    fetchData();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    setResult(null);
    e.preventDefault();

    for (let key in formData) {
      if (!formData[key]) {
        setError(`The field "${key}" is required.`);
        return;
      }
    }

    console.log("Submitting form data:", formData);

    try {
      const requestOptions = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(formData),
      };

      const response = await fetch("/api/insert_row/", requestOptions);
      const data = await response.json();

      if (data.success) {
        setResult(data.success);
        setError(null);
      } else {
        setError(data.error || "Unknown error occurred");
        setResult(null);
      }
    } catch (error) {
      console.error("Error submitting data:", error);
      setError("Error submitting data");
    }
  };

  const renderDropdownOptions = (options) => {
    try {
      const parsedOptions = JSON.parse(options);
      console.log(parsedOptions);
      return parsedOptions.map((option) => (
        <option key={option.main} value={option.main}>
          {option.main}
        </option>
      ));
    } catch (error) {
      console.error("Error parsing options:", error);
      return null;
    }
  };

  const renderFormFields = () => {
    const formFields = [];
    for (let key in formData) {
      const isNumberField = [""].includes(key);
      formFields.push(
        <div key={key}>
          <label
            htmlFor={`form-${key}`}
            className="block text-md pl-4 font-abo-one text-gray-800"
          >
            {key.charAt(0).toUpperCase() + key.slice(1)}:
          </label>
          {isNumberField ? (
            <input
              type="number"
              id={`form-${key}`}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="form-select font-reem-kufi mt-1 block w-full px-3 py-3 rounded-2xl border-lightgreen border-2 shadow-sm focus:border-mygreen"
              placeholder={`Enter ${key}`}
              title={`Enter ${key}`}
            />
          ) : (
            <select
              id={`form-${key}`}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="form-select font-reem-kufi mt-1 block w-full px-3 py-3 rounded-2xl border-lightgreen border-2 shadow-sm focus:border-mygreen"
              title={`Select ${key}`}
            >
              <option value="">Select {key}</option>
              {renderDropdownOptions(
                JSON.stringify(dropdownOptions[translator[key]] || [])
              )}
            </select>
          )}
        </div>
      );
    }
    return formFields;
  };

  // Render form once data is fetched
  return (
    <div className="max-w-lg mx-auto p-6 bg-white rounded-md">
      <h2 className="text-4xl font-abo-one text-brightgreen pb-6">
        Insert Data
      </h2>
      {loading ? (
        <div className="flex w-full pt-12 items-center justify-center">
          <svg className="pl" width="240" height="240" viewBox="0 0 240 240">
            <circle
              className="pl__ring pl__ring--a"
              cx="120"
              cy="120"
              r="105"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 660"
              strokeDashoffset="-330"
              strokeLinecap="round"
            ></circle>
            <circle
              className="pl__ring pl__ring--b"
              cx="120"
              cy="120"
              r="35"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 220"
              strokeDashoffset="-110"
              strokeLinecap="round"
            ></circle>
            <circle
              className="pl__ring pl__ring--c"
              cx="85"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 440"
              strokeLinecap="round"
            ></circle>
            <circle
              className="pl__ring pl__ring--d"
              cx="155"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              strokeWidth="20"
              strokeDasharray="0 440"
              strokeLinecap="round"
            ></circle>
          </svg>
        </div>
      ) : (
        <>
          <form onSubmit={handleSubmit} className="space-y-6">
            {renderFormFields()}
            <div className="w-1/2 mx-auto pt-4">
              <button
                type="submit"
                className="w-full font-reem-kufi text-lg text-black py-3 px-4 rounded-2xl bg-lightgreen focus:outline-none hover:bg-mygreen"
              >
                Insert Data
              </button>
            </div>
          </form>
          {result && (
            <div className="mt-4 text-green-500">
              <h3 className="text-xl font-bold">Result:</h3>
              <pre className="bg-gray-100 p-4 rounded-md">
                {JSON.stringify("Successfully inputted row", null, 2)}
              </pre>
            </div>
          )}
          {error && (
            <div className="mt-4 text-red-500">
              <h3 className="text-xl font-bold">Error:</h3>
              <pre className="bg-gray-100 p-4 rounded-md">{error}</pre>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default InsertFormRow;
