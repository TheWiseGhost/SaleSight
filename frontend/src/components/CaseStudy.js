import React from "react";

const CaseStudy = () => {
  const tableData = [
    [
      "50% Off",
      "$1,000 Website",
      "Facebook DM",
      "Facebook Group",
      "Base",
      "Said Maybe",
      "No Sale",
    ],
    [
      "50% Off",
      "$1,000 Website",
      "Instagram DM",
      "Facebook Group",
      "Base",
      "Said Yes",
      "Bought Offer",
    ],
    [
      "Birthday Sale!",
      "$1,000 Website",
      "Facebook DM",
      "Facebook Group",
      "Base",
      "Ask Spouse",
      "No Sale",
    ],
    [
      "No Offer",
      "$100 SEO Boost",
      "Zoom Webinar",
      "Cold Call",
      "Upsells",
      "Emailed Back",
      "Bought Offer",
    ],
    [
      "Free E-Book",
      "$100 SEO Boost",
      "Launch Presentation",
      "Tik Tok",
      "Upsells",
      "No Response",
      "No Sale",
    ],
    [
      "Free SEO",
      "$15,000 Full Development",
      "Zoom Webinar",
      "Cold Call",
      "Top Tier",
      "Ask Spouse",
      "No Sale",
    ],
    [
      "No Offer",
      "$15,000 Full Development",
      "Coffee Shop",
      "Cold Call",
      "Top Tier",
      "Said No",
      "No Sale",
    ],
  ];

  const tableHeaders = [
    "Offer",
    "Product",
    "Location",
    "Lead",
    "Funnel Stage",
    "Response",
    "Result",
  ];

  return (
    <div className="min-h-screen pb-12 pt-36" id="example">
      <div className="p-4 justify-center">
        <h1 className="text-4xl md:text-6xl font-abo-one text-center mb-2">
          7 <span className="text-amber-400">Essential Features</span> to Track
          your Data
        </h1>
        <p className="text-center text-xl md:text-2xl text-gray-600 font-reem-kufi mt-6 md:mt-12">
          Case Study with Amir - Web Agency Owner doing over 10,000 outreaches a
          month
        </p>
        <div className="overflow-x-auto mt-8">
          <table className="w-full md:w-11/12 mx-auto table-auto font-reem-kufi border-collapse border-2 md:border-4 border-gray-400">
            <thead>
              <tr>
                {tableHeaders.map((header, index) => (
                  <th
                    key={index}
                    className="min-w-20 md:min-w-40 border-2 md:border-4 text-xs md:text-sm md:text-md border-gray-400 px-2 md:px-4 py-1 md:py-2 font-bold text-center"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="border-2 md:border-4 text-xs md:text-sm md:text-md border-gray-400 p-1 md:p-4 text-center"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default CaseStudy;
