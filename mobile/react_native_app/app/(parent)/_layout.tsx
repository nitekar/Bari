import React from 'react';
import { Tabs } from 'expo-router';
import { Platform, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../../src/shared/theme/colors';
import { Logo } from '../../src/shared/components';

const TAB_BAR_STYLE = {
  position: 'absolute' as const,
  bottom: Platform.OS === 'ios' ? 24 : 12,
  left: 16, right: 16,
  borderRadius: 28, height: 64,
  backgroundColor: colors.surface,
  borderTopWidth: 0, elevation: 16,
  shadowColor: colors.primaryDark,
  shadowOffset: { width: 0, height: 6 },
  shadowOpacity: 0.18, shadowRadius: 16,
  paddingBottom: 0, paddingTop: 0,
};

export default function ParentLayout() {
  return (
    <Tabs screenOptions={{
      tabBarStyle: TAB_BAR_STYLE,
      tabBarActiveTintColor: colors.secondaryDark,
      tabBarInactiveTintColor: colors.textLight,
      tabBarLabelStyle: { fontSize: 10, fontWeight: '700', marginBottom: 6 },
      tabBarIconStyle: { marginTop: 6 },
      headerStyle: { backgroundColor: colors.surface, elevation: 0, shadowOpacity: 0, borderBottomWidth: 1, borderBottomColor: colors.border },
      headerTintColor: colors.secondaryDark,
      headerTitleStyle: { fontWeight: '700', fontSize: 17, color: colors.text },
      headerLeft: () => <View style={{ marginLeft: 16 }}><Logo size="sm" showText /></View>,
    }}>
      <Tabs.Screen name="index"     options={{ title: 'Home',        headerShown: false, tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'home' : 'home-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="baby"      options={{ title: 'Baby',        tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'heart' : 'heart-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="results"   options={{ title: 'Results',     tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'pulse' : 'pulse-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="education" options={{ title: 'Learn',       tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'book' : 'book-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="profile"   options={{ title: 'Profile',     tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'person' : 'person-outline'} size={22} color={color} /> }} />
    </Tabs>
  );
}
