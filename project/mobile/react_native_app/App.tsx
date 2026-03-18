/**
 * Bari Anemia Screening App
 * Root application entry point with navigation setup.
 */

import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {SafeAreaProvider} from 'react-native-safe-area-context';

import CaptureScreen from './src/screens/CaptureScreen';
import PatientInfoScreen from './src/screens/PatientInfoScreen';
import ResultScreen from './src/screens/ResultScreen';

export type RootStackParamList = {
  Capture: undefined;
  PatientInfo: {imageUri: string};
  Result: {prediction: PredictionResult};
};

export interface PredictionResult {
  diagnosis: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  nutrition_advice: string;
  recommended_foods: string[];
  referral_action: string;
  urgency: string;
}

const Stack = createNativeStackNavigator<RootStackParamList>();

function App(): React.JSX.Element {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Capture"
          screenOptions={{
            headerStyle: {backgroundColor: '#1a73e8'},
            headerTintColor: '#fff',
            headerTitleStyle: {fontWeight: 'bold'},
          }}>
          <Stack.Screen
            name="Capture"
            component={CaptureScreen}
            options={{title: 'Bari — Anemia Screening'}}
          />
          <Stack.Screen
            name="PatientInfo"
            component={PatientInfoScreen}
            options={{title: 'Patient Information'}}
          />
          <Stack.Screen
            name="Result"
            component={ResultScreen}
            options={{title: 'Screening Result'}}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

export default App;
