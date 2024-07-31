import React, { useEffect, useState } from "react";
import "../../static/css/index.css";

const FeatureDisplay = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      const token = localStorage.getItem("token"); // Retrieve the token from localStorage

      try {
        const response = await fetch("/api/features/", {
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
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const jsonData = await response.json();

        if (jsonData.warning) {
          return;
        }

        console.log("Fetched data:", jsonData.message);

        // Parsing the stringified JSON array

        console.log("Parsed data array:", parsedData);

        if (Array.isArray(parsedData)) {
          const sortedData = parsedData.sort(
            (a, b) => b.Importance - a.Importance
          );
          setData(sortedData.slice(0, 6));
        } else {
          console.error("Parsed data is not an array:", parsedData);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="overflow-x-auto">
      {loading ? (
        <div className="flex w-full pt-16 items-center justify-center">
          <svg class="pl" width="240" height="240" viewBox="0 0 240 240">
            <circle
              class="pl__ring pl__ring--a"
              cx="120"
              cy="120"
              r="105"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 660"
              stroke-dashoffset="-330"
              stroke-linecap="round"
            ></circle>
            <circle
              class="pl__ring pl__ring--b"
              cx="120"
              cy="120"
              r="35"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 220"
              stroke-dashoffset="-110"
              stroke-linecap="round"
            ></circle>
            <circle
              class="pl__ring pl__ring--c"
              cx="85"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 440"
              stroke-linecap="round"
            ></circle>
            <circle
              class="pl__ring pl__ring--d"
              cx="155"
              cy="120"
              r="70"
              fill="none"
              stroke="#000"
              stroke-width="20"
              stroke-dasharray="0 440"
              stroke-linecap="round"
            ></circle>
          </svg>
        </div>
      ) : (
        <div className="w-full items-center">
          <table className="table table-zebra mx-auto w-full">
            <thead>
              <tr className="text-lg font-normal font-abo-one">
                <th className="font-normal text-black">Feature</th>
                <th className="font-normal text-black">Importance</th>
              </tr>
            </thead>
            <tbody className="font-reem-kufi">
              {data.length > 0 ? (
                data.map((item, index) => (
                  <tr key={index}>
                    <td className="text-black">{item.Feature}</td>
                    <td className="text-gray-700">
                      {(item.Importance * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="2" className="text-center">
                    20 Rows of Data required for insights
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default FeatureDisplay;
