/**
 * app.config.js — Dynamic Expo config that reads secrets from .env
 *
 * Replaces the static app.json. Expo automatically picks this up.
 * Uses dotenv to load .env variables at build time.
 */
require('dotenv').config();

module.exports = ({ config }) => {
  return {
    ...config,
    name: "Anemia Screening",
    slug: "anemia-screening",
    version: "2.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    scheme: "anemia-screening",
    userInterfaceStyle: "light",
    newArchEnabled: true,
    splash: {
      backgroundColor: "#FFF5F0",
      resizeMode: "contain",
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#FFF5F0",
      },
      package: "com.bari.anemiascreening",
      permissions: [
        "android.permission.CAMERA",
        "android.permission.INTERNET",
      ],
      blockedPermissions: [
        "android.permission.RECORD_AUDIO",
      ],
    },
    web: {
      favicon: "./assets/favicon.png",
      bundler: "metro",
    },
    ios: {
      supportsTablet: true,
      bundleIdentifier: "com.bari.anemiascreening",
      infoPlist: {
        NSCameraUsageDescription: "This app uses the camera to photograph conjunctiva for anemia screening.",
        NSPhotoLibraryUsageDescription: "This app accesses your photo library to select conjunctiva images for anemia screening.",
      },
    },
    plugins: [
      "expo-router",
      "expo-font",
      "expo-secure-store",
      [
        "expo-image-picker",
        {
          photosPermission: "Allow access to select conjunctiva images for screening.",
          cameraPermission: "Allow camera access to photograph conjunctiva for screening.",
        },
      ],
    ],
    extra: {
      router: {},
      supabaseUrl: process.env.SUPABASE_URL || "https://YOUR_PROJECT_REF.supabase.co",
      supabaseAnonKey: process.env.SUPABASE_ANON_KEY || "YOUR_ANON_KEY",
      apiBaseUrl: process.env.API_BASE_URL || "https://YOUR_BACKEND_URL",
      apiKey: process.env.API_KEY || "",
      eas: {
        projectId: "25e00d49-367f-4807-8d33-6c6e8db1d66b",
      },
    },
    experiments: {
      typedRoutes: true,
    },
  };
};
