import React, { useState } from 'react';
import axios from 'axios';
import { usePlaceContext } from './Maps/PlaceContext';
import {useNavigate} from 'react-router-dom'

const MainDash = () => {
  const [message, setMessage] = useState('');
  const { setPlace } = usePlaceContext(); 
  const navigate = useNavigate()

  const handleSubmit = async (event) => {
    event.preventDefault();
  
    try {
      const response = await axios.post('http://localhost:5000/extract_locations', { text: message });
      const { place, latitude, longitude } = response.data;
      console.log("Response:", response.data); // Add this line to see the entire response
      console.log("Place:", place); // Add these lines to check individual values
      console.log("Latitude:", latitude);
      console.log("Longitude:", longitude);
      // Assuming your context has setPlace
      if (setPlace) {
        const extractedLocations = { place, latitude, longitude };
        setPlace(extractedLocations);
        console.log(extractedLocations)
        navigate("/final")
      } else {
        console.error('setPlace function is not defined in the context');
      }
    } catch (error) {
      console.error('Error extracting locations:', error);
    }
  };
  

  return (
    <div>
      <div className='bg-gradient-to-r from-blue-600 to-blue-200 text-white font-extrabold'>
        <p className="text-4xl px-10 pb-5 pt-5 text-center">Geolocation Extraction Tool</p>
        <p className="text-lg italic px-10 pb-5 text-center">Enter Your message to find the geolocation</p>
      </div>
      <div className='flex items-center justify-center my-10'>
        <div className="w-[85%] mb-4 border border-gray-200 rounded-lg bg-gray-50 dark:bg-gray-700 dark:border-gray-600">
          <div className="px-4 py-2 bg-white rounded-t-lg dark:bg-gray-800">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              id="comment"
              rows="10"
              className="w-full p-2 text-sm text-gray-900 bg-white border-0 dark:bg-gray-800 focus:ring-0 dark:text-white dark:placeholder-gray-400 focus:border-none"
              placeholder="Enter your message..."
              required
            ></textarea>
          </div>
          <div className="flex items-center justify-between px-3 py-2 border-t dark:border-gray-600">
            <button
              onClick={handleSubmit}
              type="submit"
              className="inline-flex items-center py-2.5 px-4 text-md font-medium text-center text-white bg-blue-700 rounded-lg focus:ring-4 focus:ring-blue-200 dark:focus:ring-blue-900 hover:bg-blue-800"
            >
              Extract Geolocations
            </button>
            <div className="flex ps-0 space-x-1 rtl:space-x-reverse sm:ps-2">
              <button
                type="button"
                className="inline-flex justify-center items-center p-2 text-gray-500 rounded cursor-pointer hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-600"
              >
                <svg
                  className="w-4 h-4"
                  aria-hidden="true"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 12 20"
                >
                  <path
                    stroke="currentColor"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M1 6v8a5 5 0 1 0 10 0V4.5a3.5 3.5 0 1 0-7 0V13a2 2 0 0 0 4 0V6"
                  />
                </svg>
                <span className="sr-only">Attach file</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainDash;
