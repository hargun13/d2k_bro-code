import React, { createContext, useState } from 'react';

export const ExerciseIDContext = createContext();

export const ExerciseIdProvider = ({ children }) => {
  const [exerciseId, setExerciseId] = useState(null);

  return (
    <ExerciseIDContext.Provider value={{ exerciseId, setExerciseId }}>
      {children}
    </ExerciseIDContext.Provider>
  );
};
