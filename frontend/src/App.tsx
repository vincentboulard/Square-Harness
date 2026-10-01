import { useEffect, useState } from 'react'
import { ApprovalDialog } from './components/Approval'
import { DropZone } from './components/DropZone'
import { Login } from './components/Login'
import { BrandMark, Rail, SideSpine, Sidebar, TopBar } from './components/Shell'
import { MODES } from './modes'
import { usePanelHidden } from './panels'
import { notebook, useRoute, type Route } from './router'
import { useApp } from './store'
import { About } from './views/About'
import { ChatView } from './views/ChatView'
import { ProofNew } from './views/ProofNew'
import { ProofView } from './views/ProofView'
import { ResearchNew } from './views/ResearchNew'
import { ResearchView } from './views/ResearchView'
import { WriteupNew } from './views/WriteupNew'
import { WriteupView } from './views/WriteupView'
import { ExperimentNew } from './views/ExperimentNew'
import { ExperimentView } from './views/ExperimentView'

function View({ route }: { route: Route }) {
  if (route.page === 'about') return <About />
  const info = MODES[route.mode]
  if (info.kind === 'proof') return route.id ? <ProofView key={route.id} id={route.id} tab={route.tab} /> : <ProofNew />
  if (info.kind === 'chat' || info.kind === 'free') {
    // Older critique and exploration chats open as they were; new ones start in Default.
    const mode = route.id ? (route.mode as 'critic' | 'explore' | 'free') : 'free'
    return <ChatView key={mode + (route.id || '')} mode={mode} id={route.id} />
  }
  if (info.kind === 'experiment') return route.id ? <ExperimentView key={route.id} id={route.id} tab={route.tab} /> : <ExperimentNew />
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
  useEffect(() => setDrawer(false), [route.mode, route.id, route.page])
  if (app.auth === 'checking') {
    return <div className="splash paper"><BrandMark size={56} /></div>
  }
  if (app.auth === 'needed') return <Login />
  const about = route.page === 'about'
  return (
    <div className={'app' + (listHidden && !about ? ' list-hidden' : '')}>
      <div className={'nav' + (drawer ? ' nav-open' : '')}>
        <Rail route={route} />
        {!about && <Sidebar route={route} />}
      </div>
      {!about && listHidden && <SideSpine route={route} />}
      {drawer && <div className="scrim" onClick={() => setDrawer(false)} />}
      <main className="main">
        <TopBar title={about ? 'About' : MODES[notebook(route.mode)].label} onMenu={() => setDrawer(true)} />
        {app.connection === 'reconnecting' && (
          <div className="connection" role="status">Reconnecting to the harness. Saved work is safe; live updates resume when it answers.</div>
        )}
        <View route={route} />
      </main>
      <ApprovalDialog />
      <DropZone />
    </div>
  )
}
