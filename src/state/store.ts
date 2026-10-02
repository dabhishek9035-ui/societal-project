import { create } from 'zustand';
import type { Horizon } from '@/data/types';

type SceneMode = 'underwater' | 'tank';
type Quality = 'high' | 'medium' | 'low';
type AppState = {
  scene: SceneMode;
  selectedDam: string;
  horizon: Horizon;
  scrollVelocity: number;
  quality: Quality;
  tankFraction: number;
  predictedFraction: number;
  showPredicted: boolean;
  setScene: (scene: SceneMode) => void;
  setSelectedDam: (id: string) => void;
  setHorizon: (horizon: Horizon) => void;
  setScrollVelocity: (velocity: number) => void;
  setQuality: (quality: Quality) => void;
  setTankLevels: (current: number, predicted: number, show: boolean) => void;
  setShowPredicted: (show: boolean) => void;
};

export const useAppStore = create<AppState>((set) => ({
  scene: 'underwater', selectedDam: 'almatti', horizon: 7, scrollVelocity: 0, quality: 'high', tankFraction: 0.72, predictedFraction: 0.76, showPredicted: false,
  setScene: (scene) => set({ scene }), setSelectedDam: (selectedDam) => set({ selectedDam }),
  setHorizon: (horizon) => set({ horizon }), setScrollVelocity: (scrollVelocity) => set({ scrollVelocity }),
  setQuality: (quality) => set({ quality }),
  setTankLevels: (tankFraction, predictedFraction, showPredicted) => set({ tankFraction, predictedFraction, showPredicted }),
  setShowPredicted: (showPredicted) => set({ showPredicted }),
}));
