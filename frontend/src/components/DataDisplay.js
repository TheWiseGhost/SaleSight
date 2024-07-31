import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import IconButton from "@mui/material/IconButton";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import DeleteIcon from "@mui/icons-material/Delete";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import InsertRowForm from "./InsertRowForm";
import Popup from "./Popup";
import { useMediaQuery } from "react-responsive";

const DataDisplay = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const token = localStorage.getItem("token");
  const navigate = useNavigate();
  const isSmallScreen = useMediaQuery({ maxWidth: 768 });

  const [isInsertPopupOpen, setIsInsertPopupOpen] = useState(false);
  const [showDeleteIcons, setShowDeleteIcons] = useState(false);

  const openInsertPopup = () => setIsInsertPopupOpen(true);
  const closeInsertPopup = () => setIsInsertPopupOpen(false);

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleDelete = async (id) => {
    try {
      console.log(id);
      const response = await fetch(`/api/delete_row/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ row: id }),
      });

      if (response.auth) {
        navigate("/auth");
      }

      if (response.ok) {
        window.location.reload();
      } else {
        console.error("Failed to delete the row.");
      }
    } catch (error) {
      console.error("Error deleting row:", error);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/data/", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        });
        const res = await response.json();
        if (res.auth) {
          navigate("/auth");
        }

        if (res.data) {
          setData(res.data);
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        try {
          setLoading(true);
          const response = await fetch("/api/data/", {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
          });
          const res = await response.json();
          if (res.auth) {
            navigate("/auth");
          }

          if (res.data) {
            setData(res.data);
          }
        } catch (error) {
          console.error("Error fetching data:", error);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchData();
  }, [token]);

  const handleExport = async () => {
    try {
      const response = await fetch("/api/export_csv/", {
        method: "GET",
        headers: {
          "Content-Type": "text/csv",
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "salesight_data.csv";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("There was an error exporting the CSV:", error);
    }
  };

  if (isSmallScreen) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <img src="../../static/images/SaleSightLogo.png" className="w-1/4" />
        <h1 className="text-5xl text-center font-reem-kufi pt-6 pb-5">
          Not configured for mobile
        </h1>
        <p className="text-xl font-reem-kufi">
          Please use SaleSight on a computer
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col mt-8 mx-10 flex-grow pb-16">
      <div className="flex items-center w-full justify-between px-12">
        <h2 className="text-4xl font-abo-one w-1/3 text-brightgreen">
          All Data
        </h2>
        <div className="w-1/3 justify-center">
          <img
            src="../../static/images/SaleSightLogo.png"
            className="w-1/6 mx-auto"
          />
        </div>
        <div className="space-x-8 w-1/3 flex flex-row justify-end">
          <button
            onClick={() => handleNavigation("/dataportal")}
            className="py-3 font-abo-one px-6 bg-mygreen hover:bg-lightgreen transition duration-200 rounded-xl"
          >
            Data Portal
          </button>
          <button
            onClick={() => handleNavigation("/dashboard")}
            className="py-3 font-abo-one px-6 bg-yellow-300 hover:bg-yellow-200 transition duration-200 rounded-xl"
          >
            Home
          </button>
        </div>
      </div>

      <div className="flex items-center pb-6 pt-10 w-full justify-between px-12">
        <div className="w-1/3 flex flex-row space-x-4 justify-start">
          <IconButton
            onClick={openInsertPopup}
            style={{
              backgroundColor: "#B2FF97",
              color: "black",
              width: "2.5rem",
              height: "2.5rem",
              marginTop: "auto",
            }}
          >
            <AddIcon fontSize="inherit" />
          </IconButton>
          <IconButton
            onClick={() => setShowDeleteIcons((prev) => !prev)}
            style={{
              backgroundColor: "#FF6C63",
              color: "black",
              width: "2.5rem",
              height: "2.5rem",
            }}
          >
            <RemoveIcon fontSize="inherit" />
          </IconButton>
        </div>
        <div className="w-1/3 justify-center">
          <h1 className="font-abo-one text-center text-black text-4xl">
            Data Table
          </h1>
        </div>
        <div className="w-1/3 flex flex-row items-baseline justify-end">
          {!loading ? (
            <IconButton
              onClick={() => handleExport()}
              style={{
                backgroundColor: "white",
                color: "black",
                width: "2.5rem",
                height: "2.5rem",
              }}
            >
              <FileDownloadIcon fontSize="large" />
            </IconButton>
          ) : (
            <></>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <table className="table-fixed w-full">
          <thead>
            <tr className="bg-white h-12 rounded-xl w-full justify-between items-center">
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Response
              </th>
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Location
              </th>
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Offer
              </th>
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Sale Product
              </th>
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Lead Gen
              </th>
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Funnel Stage
              </th>
              <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                Result
              </th>
              {showDeleteIcons && (
                <th className="text-center font-semibold border-4 border-gray-400 py-2 w-full">
                  Delete
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr className="">
                <td />
                <td />
                <td />
                <td className="flex justify-center items-center h-24">
                  <svg className="pl" viewBox="0 0 240 240">
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
                </td>
                <td />
                <td />
                <td />
              </tr>
            ) : (
              <>
                {data.map((row, index) => (
                  <tr
                    key={index}
                    className="bg-white h-12 rounded w-full justify-between px-8 items-center font-reem-kufi"
                  >
                    <td className="text-center border-2 border-gray-400">
                      {row.response}
                    </td>
                    <td className="text-center border-2 border-gray-400">
                      {row.location}
                    </td>
                    <td className="text-center border-2 border-gray-400">
                      {row.offer}
                    </td>
                    <td className="text-center border-2 border-gray-400">
                      {row.sale_product}
                    </td>
                    <td className="text-center border-2 border-gray-400">
                      {row.lead_gen}
                    </td>
                    <td className="text-center border-2 border-gray-400">
                      {row.funnel_stage}
                    </td>
                    <td className="text-center border-2 border-gray-400">
                      {row.result}
                    </td>
                    {showDeleteIcons && (
                      <td className="text-center border-2 border-gray-400">
                        <IconButton
                          onClick={() => handleDelete(row.id)}
                          style={{
                            backgroundColor: "#FF6C63",
                            color: "black",
                            width: "2rem",
                            height: "2rem",
                          }}
                        >
                          <DeleteIcon fontSize="inherit" />
                        </IconButton>
                      </td>
                    )}
                  </tr>
                ))}
              </>
            )}
          </tbody>
        </table>
      </div>
      <Popup isOpen={isInsertPopupOpen} onClose={closeInsertPopup}>
        <InsertRowForm />
      </Popup>
    </div>
  );
};

export default DataDisplay;
