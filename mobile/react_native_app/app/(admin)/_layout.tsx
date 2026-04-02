import React from 'react';
import { Tabs } from 'expo-router';
import { Platform, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../../src/shared/theme/colors';
import { Logo, LanguageSwitch } from '../../src/shared/components';

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

export default function AdminLayout() {
  return (
    <Tabs screenOptions={{
      tabBarStyle: TAB_BAR_STYLE,
      tabBarActiveTintColor: colors.primaryDark,
      tabBarInactiveTintColor: colors.textLight,
      tabBarLabelStyle: { fontSize: 10, fontWeight: '700', marginBottom: 6 },
      tabBarIconStyle: { marginTop: 6 },
      headerStyle: { backgroundColor: colors.surface, elevation: 0, shadowOpacity: 0, borderBottomWidth: 1, borderBottomColor: colors.border },
      headerTintColor: colors.primaryDark,
      headerTitleStyle: { fontWeight: '700', fontSize: 17, color: colors.text },
      headerLeft: () => <View style={{ marginLeft: 16 }}><Logo size="sm" showText /></View>,
      headerRight: () => <LanguageSwitch />,
    }}>
      <Tabs.Screen name="index"     options={{ title: 'Dashboard', headerShown: false, tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'grid' : 'grid-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="users"     options={{ title: 'Users',     tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'people' : 'people-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="analytics" options={{ title: 'Analytics', tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'bar-chart' : 'bar-chart-outline'} size={22} color={color} /> }} />
      <Tabs.Screen name="audit"     options={{ title: 'Audit',     tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'document-text' : 'document-text-outline'} size={22} color={color} /> }} />
    </Tabs>
  );
}
