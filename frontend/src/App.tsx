import { useEffect, useState } from 'react'
import { ApprovalDialog } from './components/Approval'
import { DropZone } from './components/DropZone'
import { Login } from './components/Login'
import { JobNavigation } from './components/JobNavigation'
import { BrandMark, SideSpine, Sidebar, TopBar } from './components/Shell'
import { MODES } from './modes'
import { usePanelHidden } from './panels'
import { useRoute, type Route } from './router'
import { useApp } from './store'
import { ChatView } from './views/ChatView'
import { ProofNew } from './views/ProofNew'
import { ProofView } from './views/ProofView'
import { ResearchNew } from './views/ResearchNew'
import { ResearchView } from './views/ResearchView'
import { WriteupNew } from './views/WriteupNew'
import { WriteupView } from './views/WriteupView'

function View({ route }: { route: Route }) {
  const info = MODES[route.mode]
  if (info.kind === 'proof') return route.id ? <ProofView key={route.id} id={route.id} tab={route.tab} /> : <ProofNew />
  if (info.kind === 'chat' || info.kind === 'free') {
    // Older critique and exploration chats open as they were; new ones start in Default.
    const mode = route.id ? (route.mode as 'critic' | 'explore' | 'free') : 'free'
    return <ChatView key={mode + (route.id || '')} mode={mode} id={route.id} />
  }
  if (info.kind === 'writeup') return route.id ? <WriteupView key={route.id} id={route.id} tab={route.tab} /> : <WriteupNew />
  return route.id
    ? <ResearchView key={route.id} id={route.id} tab={route.tab} />
    : <ResearchNew key={route.mode} kind={route.mode as 'literature' | 'referee'} />
}

export function App() {
  const app = useApp()
  const route = useRoute()
  const [drawer, setDrawer] = useState(false)
  const listHidden = usePanelHidden('sidebar')
  useEffect(() => setDrawer(false), [route.mode, route.id])
  if (app.auth === 'checking') {
    return <div className="splash paper"><BrandMark size={56} /></div>
  }
  if (app.auth === 'needed') return <Login />
  return (
    <div className={'app' + (listHidden ? ' list-hidden' : '')}>
      <div className={'nav' + (drawer ? ' nav-open' : '')}>
        <Sidebar route={route} />
      </div>
      {listHidden && <SideSpine route={route} />}
      {drawer && <div className="scrim" onClick={() => setDrawer(false)} />}
      <main className="main">
        <TopBar route={route} onMenu={() => setDrawer(true)} />
        {app.connection === 'reconnecting' && (
          <div className="connection" role="status">Reconnecting to the harness. Saved work is safe; live updates resume when it answers.</div>
        )}
        <JobNavigation route={route} />
        <View route={route} />
      </main>
      <ApprovalDialog />
      <DropZone />
    </div>
  )
}
