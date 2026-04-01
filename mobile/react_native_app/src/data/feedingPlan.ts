/**
 * feedingPlan.ts — Localized weekly feeding plan (Rwanda/Africa)
 */
export interface Meal { time: string; emoji: string; name: string; description: string; }
export interface DayPlan { id: string; day: string; meals: Meal[]; }

export const WEEKLY_FEEDING_PLAN: DayPlan[] = [
  { id: 'd1', day: 'Monday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Sorghum Porridge', description: 'With groundnut paste + banana' },
    { time: 'Snack', emoji: '🍌', name: 'Banana', description: 'Sliced ripe banana' },
    { time: 'Lunch', emoji: '🍛', name: 'Beans & Sweet Potato', description: 'Mashed beans, sweet potato, lemon' },
    { time: 'Snack', emoji: '🥚', name: 'Egg', description: 'Half a boiled egg' },
    { time: 'Dinner', emoji: '🍲', name: 'Isombe & Fish', description: 'Cassava leaves, small fish, ubugali' },
  ]},
  { id: 'd2', day: 'Tuesday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Millet Porridge', description: 'With milk' },
    { time: 'Snack', emoji: '🥑', name: 'Avocado', description: 'Mashed with lime' },
    { time: 'Lunch', emoji: '🍛', name: 'Rice & Lentils', description: 'Rice, lentil stew, carrot' },
    { time: 'Snack', emoji: '🍊', name: 'Orange', description: 'Fresh segments (vitamin C)' },
    { time: 'Dinner', emoji: '🍲', name: 'Fish Stew', description: 'Dried fish, tomatoes, amaranth' },
  ]},
  { id: 'd3', day: 'Wednesday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Banana Porridge', description: 'Maize + banana + groundnuts' },
    { time: 'Snack', emoji: '🥕', name: 'Carrot', description: 'Soft-cooked sticks' },
    { time: 'Lunch', emoji: '🍛', name: 'Bean Stew', description: 'Red beans, tomato, matoke' },
    { time: 'Snack', emoji: '🍈', name: 'Papaya', description: 'Ripe papaya pieces' },
    { time: 'Dinner', emoji: '🍲', name: 'Egg & Greens', description: 'Scrambled egg, spinach, ugali' },
  ]},
  { id: 'd4', day: 'Thursday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Soya Porridge', description: 'Fortified soya + pumpkin' },
    { time: 'Snack', emoji: '🥜', name: 'Groundnuts', description: 'Paste on chapati' },
    { time: 'Lunch', emoji: '🍛', name: 'Cassava & Fish', description: 'Boiled cassava, fish, veg' },
    { time: 'Snack', emoji: '🍌', name: 'Banana', description: 'Ripe banana' },
    { time: 'Dinner', emoji: '🍲', name: 'Liver & Greens', description: 'Chicken liver, amaranth, rice' },
  ]},
  { id: 'd5', day: 'Friday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Sweet Potato', description: 'Orange sweet potato + milk' },
    { time: 'Snack', emoji: '🥚', name: 'Egg', description: 'Soft-boiled egg' },
    { time: 'Lunch', emoji: '🍛', name: 'Peas & Rice', description: 'Green pea stew, rice' },
    { time: 'Snack', emoji: '🍇', name: 'Fruit Mix', description: 'Mango + watermelon' },
    { time: 'Dinner', emoji: '🍲', name: 'Meat Stew', description: 'Goat meat, potatoes, carrots' },
  ]},
  { id: 'd6', day: 'Saturday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Millet Porridge', description: 'Millet + banana' },
    { time: 'Snack', emoji: '🥑', name: 'Avocado', description: 'Mashed on bread' },
    { time: 'Lunch', emoji: '🍛', name: 'Beans & Matoke', description: 'Mixed beans, steamed banana' },
    { time: 'Snack', emoji: '🍊', name: 'Passion Fruit', description: 'Fresh juice' },
    { time: 'Dinner', emoji: '🍲', name: 'Fish & Ugali', description: 'Tilapia, kale, ugali' },
  ]},
  { id: 'd7', day: 'Sunday', meals: [
    { time: 'Breakfast', emoji: '🌅', name: 'Egg & Porridge', description: 'Scrambled egg + sorghum' },
    { time: 'Snack', emoji: '🍈', name: 'Papaya', description: 'Papaya + lime' },
    { time: 'Lunch', emoji: '🍛', name: 'Chicken Stew', description: 'Chicken, mixed veg, rice' },
    { time: 'Snack', emoji: '🥜', name: 'Groundnuts', description: 'Crushed roasted groundnuts' },
    { time: 'Dinner', emoji: '🍲', name: 'Pumpkin Soup', description: 'Pumpkin soup, bread, milk' },
  ]},
];
