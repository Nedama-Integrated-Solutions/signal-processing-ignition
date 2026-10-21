package com.nedama.signal.client83;

import com.inductiveautomation.ignition.common.script.ScriptManager;
import com.inductiveautomation.vision.api.client.AbstractClientModuleHook;
import com.nedama.signal.common83.SignalRegistration;

/** Vision client scope: client scripts. */
public class SignalClientHook extends AbstractClientModuleHook {

    @Override
    public void initializeScriptManager(ScriptManager manager) {
        super.initializeScriptManager(manager);
        SignalRegistration.registerScripts(manager);
    }
}
