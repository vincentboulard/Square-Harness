import { href, type Route } from '../router'
import { useApp } from '../store'
import { BackIcon } from './Icons'

export function JobNavigation({ route }: { route: Route }) {
  const app = useApp()
  if (!route.id || !['prove', 'literature', 'referee', 'writeup'].includes(route.mode)) return null
  const origin = route.from && (!app.chats || app.chats.some((chat) => chat.id === route.from))
    ? route.from : null
  return (
    <nav className="job-navigation" aria-label="Return to Square Harness">
      <a className="btn btn-small btn-quiet" href={href('free', origin)}>
        <BackIcon size={16} /> {origin ? 'Back to conversation' : 'Back to Square Harness'}
      </a>
    </nav>
  )
}
