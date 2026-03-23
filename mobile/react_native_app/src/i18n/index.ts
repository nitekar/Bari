/**
 * i18n/index.ts — useTranslation hook
 */
import { useStore } from '../store/useStore';
import { translations, type Language, type Translations } from './translations';

export function useTranslation(): {
  t: Translations;
  lang: Language;
  setLang: (l: Language) => void;
} {
  const language = useStore((s) => s.language);
  const setLanguage = useStore((s) => s.setLanguage);
  return {
    t: translations[language] as Translations,
    lang: language,
    setLang: setLanguage,
  };
}

export { translations, type Language, type Translations };
