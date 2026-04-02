/**
 * index.ts — Re-export all education data modules + category metadata
 */
export { MILESTONES } from './milestones';
export type { Milestone, MilestoneGroup } from './milestones';

export { FEEDING_STAGES } from './feedingStages';
export type { FeedingItem, FeedingStage } from './feedingStages';

export { NUTRIENTS, ANEMIA_SECTIONS, BABY_ACTIVITIES } from './nutrition';
export type { NutrientInfo, AnemiaSection, Activity, ActivityGroup } from './nutrition';

export { WEEKLY_FEEDING_PLAN } from './feedingPlan';
export type { Meal, DayPlan } from './feedingPlan';

export interface EducationCategory {
  id: string; title: string; subtitle: string;
  emoji: string; color: string; bg: string; route: string;
}

export const EDUCATION_CATEGORIES: EducationCategory[] = [
  { id: 'cat-milestones', title: 'Milestones', subtitle: 'Track development', emoji: '📋', color: '#7E57C2', bg: '#EDE7F6', route: '/edu-milestones' },
  { id: 'cat-feeding', title: 'Feeding Guide', subtitle: 'What to feed by age', emoji: '🥣', color: '#FF8A65', bg: '#FBE9E7', route: '/edu-feeding' },
  { id: 'cat-nutrition', title: 'Nutrition', subtitle: 'Nutrients for growth', emoji: '🥦', color: '#66BB6A', bg: '#E8F5E9', route: '/edu-nutrition' },
  { id: 'cat-anemia', title: 'Anemia Guide', subtitle: 'Causes & prevention', emoji: '🩸', color: '#EF5350', bg: '#FFEBEE', route: '/edu-anemia' },
  { id: 'cat-activities', title: 'Activities', subtitle: 'Brain-boosting play', emoji: '🧸', color: '#42A5F5', bg: '#E3F2FD', route: '/edu-activities' },
  { id: 'cat-feedingplan', title: 'Meal Plan', subtitle: 'Weekly meals', emoji: '🍽️', color: '#FFA726', bg: '#FFF3E0', route: '/edu-feedingplan' },
];
