/**
 * AnemiaScreeningApp — Root component
 * Configures React Navigation with three screens:
 *   1. CaptureScreen  — camera / gallery image selection
 *   2. PatientFormScreen — age & gender input
 *   3. ResultScreen   — prediction + nutritional guidance
 */

import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {SafeAreaProvider} from 'react-native-safe-area-context';

import CaptureScreen from './src/screens/CaptureScreen';
import PatientFormScreen from './src/screens/PatientFormScreen';
import ResultScreen from './src/screens/ResultScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Capture"
          screenOptions={{
            headerStyle: {backgroundColor: '#1a237e'},
            headerTintColor: '#fff',
            headerTitleStyle: {fontWeight: 'bold'},
          }}>
          <Stack.Screen
            name="Capture"
            component={CaptureScreen}
            options={{title: 'Anemia Screening — Capture Image'}}
          />
          <Stack.Screen
            name="PatientForm"
            component={PatientFormScreen}
            options={{title: 'Patient Information'}}
          />
          <Stack.Screen
            name="Result"
            component={ResultScreen}
            options={{title: 'Diagnosis & Guidance'}}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
