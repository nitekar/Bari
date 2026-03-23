/**
 * app/onboarding.tsx — 3-slide welcome screen for first-time users
 */
import React, { useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  ScrollView,
  TouchableOpacity,
  NativeSyntheticEvent,
  NativeScrollEvent,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius } from '../src/shared/theme';
import { Logo } from '../src/shared/components';
import { useStore } from '../src/store/useStore';
import { useTranslation } from '../src/i18n';

const { width: SCREEN_W } = Dimensions.get('window');

interface Slide {
  icon: keyof typeof Ionicons.glyphMap;
  iconBg: string;
  iconColor: string;
  titleKey: 'slide1Title' | 'slide2Title' | 'slide3Title';
  subKey: 'slide1Sub' | 'slide2Sub' | 'slide3Sub';
}

const SLIDES: Slide[] = [
  {
    icon: 'heart',
    iconBg: colors.secondary + '30',
    iconColor: colors.secondaryDark,
    titleKey: 'slide1Title',
    subKey: 'slide1Sub',
  },
  {
    icon: 'eye',
    iconBg: colors.primary + '30',
    iconColor: colors.primaryDark,
    titleKey: 'slide2Title',
    subKey: 'slide2Sub',
  },
  {
    icon: 'nutrition',
    iconBg: colors.accent + '40',
    iconColor: colors.accentDark,
    titleKey: 'slide3Title',
    subKey: 'slide3Sub',
  },
];

const BG_COLORS = [
  colors.secondary + '18',
  colors.primary + '14',
  colors.accent + '20',
];

export default function OnboardingScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const setHasSeenOnboarding = useStore((s) => s.setHasSeenOnboarding);
  const scrollRef = useRef<ScrollView>(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  const finish = () => {
    setHasSeenOnboarding(true);
    router.replace('/');
  };

  const next = () => {
    if (currentIndex < SLIDES.length - 1) {
      const nextIdx = currentIndex + 1;
      scrollRef.current?.scrollTo({ x: nextIdx * SCREEN_W, animated: true });
      setCurrentIndex(nextIdx);
    } else {
      finish();
    }
  };

  const handleScroll = (e: NativeSyntheticEvent<NativeScrollEvent>) => {
    const idx = Math.round(e.nativeEvent.contentOffset.x / SCREEN_W);
    setCurrentIndex(idx);
  };

  const isLast = currentIndex === SLIDES.length - 1;
  const bg = BG_COLORS[currentIndex];

  return (
    <View style={[styles.container, { backgroundColor: bg }]}>
      {/* ── Decorative bubbles ── */}
      <View style={styles.bubbleTL} />
      <View style={styles.bubbleBR} />

      {/* ── Top bar ── */}
      <View style={styles.topBar}>
        <Logo size="sm" showText />
        <TouchableOpacity onPress={finish} style={styles.skipBtn}>
          <Text style={styles.skipText}>{t.onboarding.skip}</Text>
        </TouchableOpacity>
      </View>

      {/* ── Slides ── */}
      <ScrollView
        ref={scrollRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onMomentumScrollEnd={handleScroll}
        scrollEventThrottle={16}
        style={styles.slides}
      >
        {SLIDES.map((slide, i) => (
          <View key={i} style={styles.slide}>
            {/* Illustration */}
            <View style={[styles.illustrationOuter, { backgroundColor: slide.iconBg }]}>
              <View style={[styles.illustrationInner, { backgroundColor: slide.iconBg }]}>
                <Ionicons name={slide.icon} size={72} color={slide.iconColor} />
              </View>
            </View>

            <Text style={styles.slideTitle}>{t.onboarding[slide.titleKey]}</Text>
            <Text style={styles.slideSub}>{t.onboarding[slide.subKey]}</Text>
          </View>
        ))}
      </ScrollView>

      {/* ── Dots ── */}
      <View style={styles.dots}>
        {SLIDES.map((_, i) => (
          <View
            key={i}
            style={[
              styles.dot,
              currentIndex === i && styles.dotActive,
            ]}
          />
        ))}
      </View>

      {/* ── CTA button ── */}
      <TouchableOpacity style={styles.cta} onPress={next} activeOpacity={0.85}>
        <Text style={styles.ctaText}>
          {isLast ? t.onboarding.getStarted : t.onboarding.next}
        </Text>
        <Ionicons
          name={isLast ? 'checkmark-circle' : 'arrow-forward-circle'}
          size={22}
          color={colors.white}
        />
      </TouchableOpacity>

      <View style={{ height: spacing.xl }} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  // ── Decorative ──
  bubbleTL: {
    position: 'absolute',
    top: -40,
    left: -40,
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: colors.primary + '18',
  },
  bubbleBR: {
    position: 'absolute',
    bottom: 80,
    right: -50,
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: colors.secondary + '18',
  },
  // ── Top bar ──
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: 56,
    paddingBottom: spacing.md,
  },
  skipBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  skipText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  // ── Slides ──
  slides: { flex: 1 },
  slide: {
    width: SCREEN_W,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.xl,
  },
  illustrationOuter: {
    width: 180,
    height: 180,
    borderRadius: 90,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xl,
  },
  illustrationInner: {
    width: 140,
    height: 140,
    borderRadius: 70,
    alignItems: 'center',
    justifyContent: 'center',
  },
  slideTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: colors.text,
    textAlign: 'center',
    lineHeight: 36,
    marginBottom: spacing.md,
  },
  slideSub: {
    fontSize: 15,
    color: colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
  // ── Dots ──
  dots: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
    marginBottom: spacing.xl,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.border,
  },
  dotActive: {
    width: 24,
    backgroundColor: colors.primary,
  },
  // ── CTA ──
  cta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.primary,
    marginHorizontal: spacing.lg,
    paddingVertical: spacing.md + 2,
    borderRadius: borderRadius.pill,
    elevation: 4,
    shadowColor: colors.primaryDark,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  ctaText: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.white,
  },
});
