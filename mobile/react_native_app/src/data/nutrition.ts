/**
 * nutrition.ts — Nutrients, anemia education, and baby activities
 */

export interface NutrientInfo { id: string; nutrient: string; emoji: string; role: string; sources: string; }
export const NUTRIENTS: NutrientInfo[] = [
  { id: 'n-iron', nutrient: 'Iron', emoji: '🩸', role: 'Builds hemoglobin, carries oxygen to brain. Critical for cognitive development.', sources: 'Liver, beans, dark greens, eggs, small fish' },
  { id: 'n-zinc', nutrient: 'Zinc', emoji: '🛡️', role: 'Supports immunity, wound healing, cell growth.', sources: 'Meat, beans, seeds, dairy' },
  { id: 'n-vita', nutrient: 'Vitamin A', emoji: '🥕', role: 'Essential for vision, immunity, cell growth.', sources: 'Sweet potatoes, carrots, mangoes, liver, dark greens' },
  { id: 'n-vitc', nutrient: 'Vitamin C', emoji: '🍊', role: 'Boosts iron absorption 3x. Strengthens immunity.', sources: 'Oranges, tomatoes, guava, passion fruit, papaya' },
  { id: 'n-protein', nutrient: 'Protein', emoji: '💪', role: 'Builds muscle, organs, enzymes. Essential for growth.', sources: 'Eggs, fish, beans, groundnuts, milk, meat' },
  { id: 'n-folate', nutrient: 'Folate', emoji: '🧬', role: 'DNA synthesis and red blood cell production.', sources: 'Dark greens, beans, lentils, avocado' },
  { id: 'n-calcium', nutrient: 'Calcium', emoji: '🦴', role: 'Strong bones and teeth. Nerve function.', sources: 'Milk, yoghurt, small dried fish, fortified soya' },
];

export interface AnemiaSection { id: string; title: string; emoji: string; points: string[]; }
export const ANEMIA_SECTIONS: AnemiaSection[] = [
  { id: 'ae-what', title: 'What is Anemia?', emoji: '🩺', points: ['Blood has too few red blood cells to carry oxygen.', 'Iron-deficiency most common in children under 5.', 'WHO: 42% of under-5s have anemia globally.', 'Sub-Saharan Africa has the highest rates.'] },
  { id: 'ae-causes', title: 'Common Causes', emoji: '🔍', points: ['Low iron intake in daily diet.', 'Poor absorption — tea/milk blocks iron.', 'Infections: malaria, hookworm, diarrhea.', 'Rapid growth needs more iron than adults.'] },
  { id: 'ae-symptoms', title: 'Signs to Watch', emoji: '⚠️', points: ['Pale eyelids, palms, nail beds.', 'Tiredness, weakness, irritability.', 'Poor appetite, slow weight gain.', 'Frequent infections.', 'Delayed development.'] },
  { id: 'ae-prevention', title: 'Prevention', emoji: '🛡️', points: ['Iron-rich foods daily: liver, beans, greens.', 'Pair iron + vitamin C for 3x absorption.', 'Avoid tea/coffee during meals.', 'Deworm every 6 months.', 'Screen every 3–6 months.'] },
];

export interface Activity { id: string; title: string; emoji: string; description: string; }
export interface ActivityGroup { id: string; ageRange: string; emoji: string; subtitle: string; activities: Activity[]; }
export const BABY_ACTIVITIES: ActivityGroup[] = [
  { id: 'act-0-6', ageRange: '0–6 months', emoji: '🧒', subtitle: 'Sensory & bonding', activities: [
    { id: 'a1', title: 'Tummy Time', emoji: '🤸', description: '3–5 min on tummy for neck strength.' },
    { id: 'a2', title: 'Eye Tracking', emoji: '👀', description: 'Move toy across baby\'s vision.' },
    { id: 'a3', title: 'Singing', emoji: '🎵', description: 'Sing to build language foundations.' },
    { id: 'a4', title: 'Gentle Massage', emoji: '🤲', description: 'Stroke limbs for sensory awareness.' },
    { id: 'a5', title: 'Contrast Cards', emoji: '🃏', description: 'Black-white patterns for vision.' },
  ]},
  { id: 'act-6-12', ageRange: '6–12 months', emoji: '🧸', subtitle: 'Exploration & motor', activities: [
    { id: 'a6', title: 'Crawling Games', emoji: '🐛', description: 'Place toys out of reach.' },
    { id: 'a7', title: 'Stack & Knock', emoji: '🧱', description: 'Stack blocks, let baby knock.' },
    { id: 'a8', title: 'Peek-a-Boo', emoji: '🙈', description: 'Teaches object permanence.' },
    { id: 'a9', title: 'Texture Play', emoji: '🪶', description: 'Touch different surfaces.' },
    { id: 'a10', title: 'Container Play', emoji: '📦', description: 'In/out for problem-solving.' },
  ]},
  { id: 'act-1-3', ageRange: '1–3 years', emoji: '🎨', subtitle: 'Language & creativity', activities: [
    { id: 'a11', title: 'Block Towers', emoji: '🏗️', description: 'Fine motor skills + patience.' },
    { id: 'a12', title: 'Walking', emoji: '🚶', description: 'Hold hands and explore.' },
    { id: 'a13', title: 'Naming Game', emoji: '🏷️', description: 'Point and name for vocabulary.' },
    { id: 'a14', title: 'Scribbling', emoji: '🖍️', description: 'Pre-writing with crayons.' },
    { id: 'a15', title: 'Dance', emoji: '💃', description: 'Music for coordination.' },
    { id: 'a16', title: 'Sorting', emoji: '🔵', description: 'Colour/size for logic.' },
  ]},
];
