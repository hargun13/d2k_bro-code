import React, { createContext, useState, useContext } from 'react';

// Create a context
const PlaceContext = createContext();

// Create a custom hook to use the context
export const usePlaceContext = () => useContext(PlaceContext);

// Create a context provider component
export const AnalysisProvider = ({ children }) => {
  const [place, setPlace] = useState(null);

  return (
    <PlaceContext.Provider value={{ place, setPlace }}>
      {children}
    </PlaceContext.Provider>
  );
};