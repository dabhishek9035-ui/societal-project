'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search, SlidersHorizontal } from 'lucide-react';
import { fetchDams } from '@/data/api';
import { DamCard, Eyebrow, PageHeading } from '@/components/Ui';

export default function DamsPage() {
  const [search,setSearch]=useState('');
  const searchRef=useRef<HTMLInputElement>(null);
  const {data:dams=[],isLoading,isError,refetch}=useQuery({queryKey:['dams'],queryFn:({signal})=>fetchDams(signal)});
  const filtered=useMemo(()=>dams.filter((dam)=>`${dam.name} ${dam.reservoir} ${dam.location} ${dam.river} ${dam.district}`.toLowerCase().includes(search.toLowerCase())),[dams,search]);
  useEffect(()=>{const onKey=(event:KeyboardEvent)=>{if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();searchRef.current?.focus();}};window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey);},[]);
  return <section className="catalog-page">
    <div className="catalog-top"><PageHeading label="THE RESERVOIR NETWORK" title={<>A closer look at<br/><span>the waterline.</span></>} description="Ten reservoirs. Distinct river basins. One shared picture of how water is moving across Karnataka." count={`${dams.length || 10} monitored`} /><div className="catalog-mark"><SlidersHorizontal size={16}/><span>OBSERVATIONS<br/>UPDATED DAILY</span></div></div>
    <div className="catalog-toolbar"><div className="toolbar-note"><Eyebrow>RESERVOIR REGISTER</Eyebrow><span>Browse current levels and overflow areas</span></div><label className="search-box"><Search size={16}/><input ref={searchRef} value={search} onChange={(e)=>setSearch(e.target.value)} placeholder="Search name, river or district" aria-label="Search dams by name, river or district"/><kbd>⌘ K</kbd></label></div>
    {isLoading && <div className="loading-grid">{Array.from({length:6},(_,i)=><div className="glass-card skeleton-card" key={i}><span/><span/><span/></div>)}</div>}
    {isError && <div className="glass-card error-card"><strong>Reservoir readings are unavailable.</strong><p>We couldn’t load the dam register. Check the data connection and try again.</p><button className="button button-ghost" onClick={()=>refetch()}>Retry ↻</button></div>}
    {!isLoading&&!isError&&filtered.length===0&&<div className="glass-card empty-card"><strong>No reservoirs match “{search}”.</strong><p>Try a dam name, river basin, or district.</p></div>}
    {!isLoading&&!isError&&filtered.length>0&&<div className="dam-grid">{filtered.map((dam,index)=><DamCard dam={dam} index={dams.indexOf(dam)} key={dam.id}/>)}</div>}
    <div className="catalog-foot"><span><i/> DATASET COVERAGE · {dams.length || 10} RESERVOIRS</span><span>SELECT A FORECAST TO EXPLORE TIME TRAVEL <b>↗</b></span></div>
  </section>;
}
