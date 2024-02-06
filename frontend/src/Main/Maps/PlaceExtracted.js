import React, { useRef, useState } from "react";
import { useNavigate } from "react-router";
import {Link} from 'react-router-dom';
import {
    useJsApiLoader,
    GoogleMap,
    MarkerF,
    MarkerClustererF,
    Autocomplete,
    DirectionsRenderer,
    InfoWindowF,
  } from "@react-google-maps/api";
import schoolsData from './HospitalsData';
import Button from '@mui/material/Button';
// import {useHospitalContext} from './hospitalContext'

const center = { lat: 19.11, lng: 72.84 };

const NearbyHospitals = () => {

    const { isLoaded } = useJsApiLoader({
        googleMapsApiKey: "AIzaSyB2zhvHVb_IKgDBRMCwr3-taL8K8lK8-90",
        libraries: ["places"],
      });

      // const {setHospital} = useHospitalContext();
    
      const [dirRes, setDirRes] = useState(null);
      const [isOpen, setisOpen] = useState(false);
      const [distance, setDistance] = useState("");
      const [duration, setDuration] = useState("");
    
      /** @type React.MutableRefObject<HTMLInputElement> */
      const originRef = useRef();
      /** @type React.MutableRefObject<HTMLInputElement> */
      const destRef = useRef();
    
      async function calculate() {
        // eslint-disable-next-line no-undef
        const directService = new google.maps.DirectionsService();
        const results = await directService.route({
          origin: originRef.current.value,
          destination: destRef.current.value,
          // eslint-disable-next-line no-undef
          travelMode: google.maps.TravelMode.DRIVING,
        });
    
        setDirRes(results);
        setDistance(results.routes[0].legs[0].distance.text);
        setDuration(results.routes[0].legs[0].duration.text);
      }
    
      const locations = schoolsData.results;
    
      const onLoad = (infoWindow) => {
        console.log("infoWindow: ", infoWindow);
      };

      // const handleHospitalName = (hospitalName) => {
      //       setHospital(hospitalName)
      // }
    
      if (isLoaded)
        return (
          <div className="h-[80vh] w-full ">
            <div className="px-10 py-4 bg-gradient-to-r from-blue-600 to-blue-200 text-white text-xl pl-16 font-bold ">
              Place Extracted
            </div>
            <div className="bg-slate-500 h-full w-full">
              <GoogleMap
                center={center}
                zoom={15}
                mapContainerStyle={{ width: "100%", height: "100%" }}
              >
                <MarkerClustererF>
                  {(clusterer) => {
                    return locations.map((location, idx) => {
                      return (
                        <MarkerF
                          key={idx}
                          position={location.geometry.location}
                          clusterer={clusterer}
                          onClick={() => setisOpen(idx)}
                        >
                          {isOpen === idx && (
                            <InfoWindowF onCloseClick={() => setisOpen(null)}>
                              <div>hello</div>
                            </InfoWindowF>
                          )}
                        </MarkerF>
                      );
                    });
                  }}
                </MarkerClustererF>
                {dirRes && <DirectionsRenderer directions={dirRes} />}
              </GoogleMap>
            </div>
          </div>
        );
}

export default NearbyHospitals