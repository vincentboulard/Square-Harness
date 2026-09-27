// A small stroke icon set; everything else in the interface is typographic.
import type { ReactNode } from 'react'

type IconProps = { size?: number; title?: string }

function Icon({ size = 18, title, children }: IconProps & { children: ReactNode }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden={title ? undefined : true} role={title ? 'img' : undefined}>
      {title && <title>{title}</title>}
      {children}
    </svg>
  )
}

export const MenuIcon = (p: IconProps) => <Icon {...p}><path d="M4 7h16M4 12h16M4 17h16" /></Icon>
export const PlusIcon = (p: IconProps) => <Icon {...p}><path d="M12 5v14M5 12h14" /></Icon>
export const PauseIcon = (p: IconProps) => <Icon {...p}><path d="M9 6v12M15 6v12" /></Icon>
export const PlayIcon = (p: IconProps) => <Icon {...p}><path d="M8 5.5v13l10-6.5z" /></Icon>
export const CloseIcon = (p: IconProps) => <Icon {...p}><path d="M6 6l12 12M18 6L6 18" /></Icon>
export const ChevronIcon = (p: IconProps) => <Icon {...p}><path d="M9 6l6 6-6 6" /></Icon>
export const SendIcon = (p: IconProps) => <Icon {...p}><path d="M5 12h13M13 6l6 6-6 6" /></Icon>
export const SunIcon = (p: IconProps) => <Icon {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" /></Icon>
export const MoonIcon = (p: IconProps) => <Icon {...p}><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" /></Icon>
export const AutoIcon = (p: IconProps) => <Icon {...p}><circle cx="12" cy="12" r="8" /><path d="M12 4v16" /><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor" /></Icon>
export const FileIcon = (p: IconProps) => <Icon {...p}><path d="M7 3h7l4 4v14H7z" /><path d="M14 3v4h4" /></Icon>
export const PinIcon = (p: IconProps) => <Icon {...p}><path d="M9 4h6l-1 6 3 3H7l3-3z" /><path d="M12 13v7" /></Icon>
export const CheckIcon = (p: IconProps) => <Icon {...p}><path d="M5 12.5l4.5 4.5L19 7" /></Icon>
export const PhoneIcon = (p: IconProps) => <Icon {...p}><rect x="7" y="2.5" width="10" height="19" rx="2" /><path d="M11 18.5h2" /></Icon>
export const BackIcon = (p: IconProps) => <Icon {...p}><path d="M15 6l-6 6 6 6" /></Icon>
// A panel with its edge: the side a panel slides to (left list, right overview).
export const PanelLeftIcon = (p: IconProps) => <Icon {...p}><rect x="3.5" y="4.5" width="17" height="15" rx="2" /><path d="M9.5 4.5v15" /><path d="M16 10l-2 2 2 2" /></Icon>
export const PanelRightIcon = (p: IconProps) => <Icon {...p}><rect x="3.5" y="4.5" width="17" height="15" rx="2" /><path d="M14.5 4.5v15" /><path d="M8 10l2 2-2 2" /></Icon>
