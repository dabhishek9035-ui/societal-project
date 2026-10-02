'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import SceneCanvas from './SceneCanvas';
import { useAppStore } from '@/state/store';

const links = [{ href: '/', label: 'Home' }, { href: '/dams', label: 'Dams' }, { href: '/predictions', label: 'Predictions' }];

export default function SiteShell({ children }: { children: React.ReactNode }) {
  const path = usePathname(); const setScene = useAppStore((s) => s.setScene); const [menuOpen,setMenuOpen]=useState(false);
  useEffect(() => { setScene(path === '/predictions' ? 'tank' : 'underwater'); setMenuOpen(false); }, [path,setScene]);
  useEffect(() => { const close=(e:KeyboardEvent)=>{if(e.key==='Escape')setMenuOpen(false);};window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close); },[]);
  return <>
    <SceneCanvas />
    <header className="site-header">
      <Link href="/" className="brand" aria-label="Jaladrishti home"><span className="brand-mark"><span /></span><span className="brand-copy"><strong>jaladrishti</strong><small>WATER INTELLIGENCE</small></span></Link>
      <nav className="desktop-nav" aria-label="Main navigation">{links.map((link)=><Link key={link.href} href={link.href} className={`nav-link ${path===link.href?'active':''}`}>{link.label}</Link>)}</nav>
      <div className="header-status"><span className="status-dot" /> LIVE OBSERVATIONS</div>
      <button type="button" className={`menu-toggle ${menuOpen?'open':''}`} aria-label={menuOpen?'Close navigation':'Open navigation'} aria-expanded={menuOpen} onClick={()=>setMenuOpen((open)=>!open)}><i/><i/><i/></button>
    </header>
    {menuOpen && <div className="mobile-drawer"><nav aria-label="Mobile navigation">{links.map((link)=><Link key={link.href} href={link.href} className={path===link.href?'active':''}>{link.label}<span>↗</span></Link>)}<p>Hydrological intelligence for Karnataka</p></nav></div>}
    <main className={`page-shell page-${path==='/predictions'?'predictions':path==='/dams'?'dams':'home'}`}>{children}</main>
    <div className="scene-caption"><span className="caption-line" /> {path==='/predictions'?'RESERVOIR CROSS-SECTION':'UNDERWATER OBSERVATORY'}</div>
  </>;
}
