/**
 * milestones.ts — Child development milestones by age group
 */
export interface Milestone { id: string; label: string; category: 'motor' | 'cognitive' | 'social'; }
export interface MilestoneGroup { id: string; ageRange: string; emoji: string; milestones: Milestone[]; }

export const MILESTONES: MilestoneGroup[] = [
  { id: 'ms-0-6', ageRange: '0–6 months', emoji: '👶', milestones: [
    { id: 'ms-0-6-1', label: 'Lifts head during tummy time', category: 'motor' },
    { id: 'ms-0-6-2', label: 'Rolls from tummy to back', category: 'motor' },
    { id: 'ms-0-6-3', label: 'Reaches for objects', category: 'motor' },
    { id: 'ms-0-6-4', label: 'Follows objects with eyes', category: 'cognitive' },
    { id: 'ms-0-6-5', label: 'Responds to sounds', category: 'cognitive' },
    { id: 'ms-0-6-6', label: 'Babbles and coos', category: 'cognitive' },
    { id: 'ms-0-6-7', label: 'Smiles at familiar faces', category: 'social' },
    { id: 'ms-0-6-8', label: 'Recognises parents', category: 'social' },
  ]},
  { id: 'ms-6-12', ageRange: '6–12 months', emoji: '🧒', milestones: [
    { id: 'ms-6-12-1', label: 'Sits without support', category: 'motor' },
    { id: 'ms-6-12-2', label: 'Crawls and pulls to stand', category: 'motor' },
    { id: 'ms-6-12-3', label: 'Uses pincer grasp', category: 'motor' },
    { id: 'ms-6-12-4', label: 'Understands "no"', category: 'cognitive' },
    { id: 'ms-6-12-5', label: 'Looks for hidden objects', category: 'cognitive' },
    { id: 'ms-6-12-6', label: 'Says first words', category: 'cognitive' },
    { id: 'ms-6-12-7', label: 'Waves bye-bye', category: 'social' },
    { id: 'ms-6-12-8', label: 'Plays peek-a-boo', category: 'social' },
  ]},
  { id: 'ms-1-2', ageRange: '1–2 years', emoji: '🚶', milestones: [
    { id: 'ms-1-2-1', label: 'Walks independently', category: 'motor' },
    { id: 'ms-1-2-2', label: 'Stacks 2–4 blocks', category: 'motor' },
    { id: 'ms-1-2-3', label: 'Scribbles with crayons', category: 'motor' },
    { id: 'ms-1-2-4', label: 'Points to body parts', category: 'cognitive' },
    { id: 'ms-1-2-5', label: 'Follows simple instructions', category: 'cognitive' },
    { id: 'ms-1-2-6', label: 'Uses 10–50 words', category: 'cognitive' },
    { id: 'ms-1-2-7', label: 'Plays pretend', category: 'social' },
    { id: 'ms-1-2-8', label: 'Shows affection', category: 'social' },
  ]},
  { id: 'ms-3-6', ageRange: '3–6 years', emoji: '🎒', milestones: [
    { id: 'ms-3-6-1', label: 'Runs, jumps, climbs well', category: 'motor' },
    { id: 'ms-3-6-2', label: 'Draws circles and shapes', category: 'motor' },
    { id: 'ms-3-6-3', label: 'Dresses self', category: 'motor' },
    { id: 'ms-3-6-4', label: 'Counts to 10+', category: 'cognitive' },
    { id: 'ms-3-6-5', label: 'Speaks in sentences', category: 'cognitive' },
    { id: 'ms-3-6-6', label: 'Knows colours/shapes', category: 'cognitive' },
    { id: 'ms-3-6-7', label: 'Takes turns and shares', category: 'social' },
    { id: 'ms-3-6-8', label: 'Plays cooperatively', category: 'social' },
  ]},
];
