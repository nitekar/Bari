/**
 * feedingStages.ts — Feeding guide by age group
 */
export interface FeedingItem { id: string; food: string; emoji: string; }
export interface FeedingStage { id: string; ageRange: string; emoji: string; texture: string; frequency: string; tips: string; foods: FeedingItem[]; }

export const FEEDING_STAGES: FeedingStage[] = [
  { id: 'fs-4-6', ageRange: '4–6 months', emoji: '🍼', texture: 'Thin purées + breastmilk', frequency: '1–2 times/day + breastmilk', tips: 'Start with single-ingredient purées. Wait 3 days between new foods.', foods: [
    { id: 'fs-4-6-1', food: 'Sweet potato purée', emoji: '🍠' },
    { id: 'fs-4-6-2', food: 'Banana mash', emoji: '🍌' },
    { id: 'fs-4-6-3', food: 'Rice cereal', emoji: '🍚' },
    { id: 'fs-4-6-4', food: 'Avocado purée', emoji: '🥑' },
    { id: 'fs-4-6-5', food: 'Pumpkin purée', emoji: '🎃' },
  ]},
  { id: 'fs-6-9', ageRange: '6–9 months', emoji: '🥣', texture: 'Thick purées + soft mashed', frequency: '2–3 times/day + breastmilk', tips: 'Introduce iron-rich foods. Combine with vitamin C.', foods: [
    { id: 'fs-6-9-1', food: 'Mashed beans', emoji: '🫘' },
    { id: 'fs-6-9-2', food: 'Egg yolk', emoji: '🥚' },
    { id: 'fs-6-9-3', food: 'Mashed cassava', emoji: '🥔' },
    { id: 'fs-6-9-4', food: 'Soft cooked greens', emoji: '🥬' },
    { id: 'fs-6-9-5', food: 'Sorghum porridge', emoji: '🥣' },
    { id: 'fs-6-9-6', food: 'Mashed papaya', emoji: '🍈' },
  ]},
  { id: 'fs-9-12', ageRange: '9–12 months', emoji: '🥄', texture: 'Chopped + finger foods', frequency: '3 meals + 1–2 snacks/day', tips: 'Small soft pieces for self-feeding. Varied textures.', foods: [
    { id: 'fs-9-12-1', food: 'Soft fish pieces', emoji: '🐟' },
    { id: 'fs-9-12-2', food: 'Groundnut paste', emoji: '🥜' },
    { id: 'fs-9-12-3', food: 'Chapati strips', emoji: '🫓' },
    { id: 'fs-9-12-4', food: 'Soft fruit pieces', emoji: '🍇' },
    { id: 'fs-9-12-5', food: 'Millet porridge', emoji: '🌾' },
    { id: 'fs-9-12-6', food: 'Mashed potatoes', emoji: '🥔' },
  ]},
  { id: 'fs-12+', ageRange: '12+ months', emoji: '🍽️', texture: 'Family foods, cut small', frequency: '3 meals + 2 snacks/day', tips: 'Most family foods OK. Avoid excess sugar and salt.', foods: [
    { id: 'fs-12-1', food: 'Rice and beans', emoji: '🍛' },
    { id: 'fs-12-2', food: 'Meat stew + veg', emoji: '🍲' },
    { id: 'fs-12-3', food: 'Matoke (banana)', emoji: '🍌' },
    { id: 'fs-12-4', food: 'Eggs', emoji: '🥚' },
    { id: 'fs-12-5', food: 'Fresh fruits', emoji: '🍊' },
    { id: 'fs-12-6', food: 'Isombe (cassava leaves)', emoji: '🥬' },
  ]},
];
