/**
 * Project icons — Lucide, with a shared stroke weight so they match the UI.
 * https://lucide.dev
 */
import {
  ArrowLeft,
  ChevronDown as ChevronDownIcon,
  CircleHelp,
  History,
  Search,
  Settings,
  X,
} from 'lucide-react';

function lucide(Icon, { strokeWidth = 1.5, ...rest } = {}) {
  return function LucideIcon({ width, height, size, ...props }) {
    return (
      <Icon
        size={size ?? width ?? height ?? 24}
        strokeWidth={strokeWidth}
        aria-hidden
        {...rest}
        {...props}
      />
    );
  };
}

export const HistoryIcon = lucide(History);
export const SearchIcon = lucide(Search);
export const SettingsIcon = lucide(Settings);
export const ClearIcon = lucide(X, { strokeWidth: 2 });
export const BackIcon = lucide(ArrowLeft);
export const GuideIcon = lucide(CircleHelp);
export const ChevronDown = lucide(ChevronDownIcon);
